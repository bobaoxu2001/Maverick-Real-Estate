"""
NYC CRE Investment Intelligence Dashboard
==========================================
Interactive Streamlit dashboard for exploring NYC commercial real estate
investment analytics — property valuations, market segments, risk scores,
time series forecasts, and Monte Carlo simulations.

Launch: streamlit run app/dashboard.py

Author: Allen Xu
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from streamlit_folium import st_folium
from pathlib import Path
import sys

# Project imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import DATA_RAW, DATA_PROCESSED, NYC_CENTER, BOROUGH_CODES

# ─────────────────────────────────────────────────────────
# Page Configuration
# ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NYC CRE Investment Intelligence",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1a1a2e;
        margin-bottom: 0;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #6c757d;
        margin-top: -10px;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 12px;
        color: white;
    }
    .stMetric > div {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #667eea;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────
# Data Loading
# ─────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    """Load processed data or generate sample data for demonstration."""
    data = {}

    # Try loading from processed directory
    for name in ["property_valuations", "market_segments", "macro_economic"]:
        path = DATA_PROCESSED / f"{name}.parquet"
        if path.exists():
            data[name] = pd.read_parquet(path)

    # If no processed data, try raw
    if not data:
        for path in DATA_RAW.glob("*.parquet"):
            data[path.stem] = pd.read_parquet(path)

    # If still no data, generate realistic demo data
    if not data:
        data = generate_demo_data()

    return data


def generate_demo_data():
    """Generate realistic demo data based on actual NYC CRE market characteristics."""
    np.random.seed(42)
    n = 5000

    boroughs = np.random.choice(
        ["Manhattan", "Brooklyn", "Queens", "Bronx", "Staten Island"],
        n,
        p=[0.30, 0.25, 0.20, 0.15, 0.10],
    )

    # Manhattan neighborhoods
    manhattan_hoods = ["Midtown", "FiDi", "Chelsea", "SoHo", "Tribeca", "UES", "UWS", "Harlem", "East Village"]
    brooklyn_hoods = ["DUMBO", "Williamsburg", "Downtown BK", "Park Slope", "Bushwick", "Red Hook"]
    queens_hoods = ["LIC", "Astoria", "Flushing", "Jamaica", "Forest Hills"]
    bronx_hoods = ["South Bronx", "Fordham", "Pelham Bay", "Riverdale"]
    si_hoods = ["St. George", "Stapleton", "Tottenville"]

    hood_map = {
        "Manhattan": manhattan_hoods,
        "Brooklyn": brooklyn_hoods,
        "Queens": queens_hoods,
        "Bronx": bronx_hoods,
        "Staten Island": si_hoods,
    }

    neighborhoods = [np.random.choice(hood_map[b]) for b in boroughs]

    # Price per sqft varies by borough/neighborhood
    base_ppsf = {
        "Manhattan": 1200, "Brooklyn": 650, "Queens": 450,
        "Bronx": 300, "Staten Island": 250,
    }

    ppsf = np.array([base_ppsf[b] * np.random.lognormal(0, 0.3) for b in boroughs])
    sqft = np.random.lognormal(np.log(15000), 0.8, n).clip(1000, 500000)
    sale_prices = ppsf * sqft

    # Coordinates (approximate NYC bounds by borough)
    lat_centers = {"Manhattan": 40.76, "Brooklyn": 40.68, "Queens": 40.73, "Bronx": 40.85, "Staten Island": 40.58}
    lon_centers = {"Manhattan": -73.97, "Brooklyn": -73.96, "Queens": -73.82, "Bronx": -73.87, "Staten Island": -74.15}

    lats = [lat_centers[b] + np.random.normal(0, 0.02) for b in boroughs]
    lons = [lon_centers[b] + np.random.normal(0, 0.02) for b in boroughs]

    dates = pd.date_range("2019-01-01", "2024-12-31", periods=n)
    year_built = np.random.randint(1890, 2024, n)
    num_floors = np.random.lognormal(np.log(8), 0.6, n).clip(1, 80).astype(int)

    # Distress probability
    building_age = 2024 - year_built
    dist_to_cbd = np.sqrt((np.array(lats) - 40.7527)**2 + (np.array(lons) + 73.9772)**2) * 111
    distress_prob = (
        0.1
        + 0.3 * (building_age > 60).astype(float)
        + 0.2 * (dist_to_cbd > 5).astype(float)
        - 0.15 * (ppsf > np.median(ppsf)).astype(float)
        + np.random.normal(0, 0.1, n)
    ).clip(0, 1)

    property_df = pd.DataFrame({
        "borough_name": boroughs,
        "neighborhood": neighborhoods,
        "sale_price": sale_prices,
        "price_per_sqft": ppsf,
        "gross_square_feet": sqft,
        "sale_date": dates,
        "sale_year": dates.year,
        "year_built": year_built,
        "building_age": building_age,
        "num_floors": num_floors,
        "latitude": lats,
        "longitude": lons,
        "dist_to_cbd_km": dist_to_cbd,
        "floor_area_ratio": np.random.lognormal(np.log(5), 0.5, n).clip(0.5, 25),
        "assessed_total": sale_prices * np.random.uniform(0.3, 0.6, n),
        "distress_probability": distress_prob,
        "risk_tier": pd.cut(distress_prob, bins=[0, 0.15, 0.35, 0.60, 1.0],
                           labels=["Low Risk", "Moderate Risk", "Elevated Risk", "High Risk"]),
        "cluster": np.random.randint(0, 6, n),
    })

    # Market segments
    market_df = property_df.groupby("neighborhood").agg({
        "sale_price": ["mean", "median", "count"],
        "price_per_sqft": ["mean", "median"],
        "building_age": "mean",
        "dist_to_cbd_km": "mean",
        "latitude": "mean",
        "longitude": "mean",
    }).reset_index()
    market_df.columns = [
        "neighborhood", "avg_price", "median_price", "n_transactions",
        "avg_ppsf", "median_ppsf", "avg_building_age", "avg_dist_cbd",
        "centroid_lat", "centroid_lon",
    ]

    # Macro data
    dates_macro = pd.date_range("2010-01-01", "2024-12-01", freq="MS")
    macro_df = pd.DataFrame({
        "date": dates_macro,
        "fed_funds_rate": np.concatenate([
            np.linspace(0.1, 0.1, 60),     # 2010-2014
            np.linspace(0.1, 2.5, 48),     # 2015-2018
            np.linspace(2.5, 0.1, 24),     # 2019-2020
            np.linspace(0.1, 5.3, 36),     # 2021-2023
            np.linspace(5.3, 4.5, 12),     # 2024
        ]),
        "treasury_10y": np.concatenate([
            np.linspace(3.5, 2.0, 60),
            np.linspace(2.0, 3.0, 48),
            np.linspace(3.0, 0.7, 24),
            np.linspace(0.7, 4.5, 36),
            np.linspace(4.5, 4.0, 12),
        ]),
        "cpi_urban": np.linspace(220, 315, len(dates_macro)) + np.random.normal(0, 2, len(dates_macro)),
        "unemployment_nyc": np.concatenate([
            np.linspace(9.0, 5.0, 60),
            np.linspace(5.0, 4.0, 48),
            np.linspace(4.0, 16.0, 12),
            np.linspace(16.0, 5.0, 24),
            np.linspace(5.0, 4.5, 24),
            np.linspace(4.5, 4.8, 12),
        ]),
    }).set_index("date")

    return {
        "property_valuations": property_df,
        "market_segments": market_df,
        "macro_economic": macro_df,
    }


# ─────────────────────────────────────────────────────────
# Sidebar Navigation
# ─────────────────────────────────────────────────────────
def sidebar():
    st.sidebar.markdown("## 🏢 NYC CRE Intelligence")
    st.sidebar.markdown("*Maverick Real Estate Partners*")
    st.sidebar.markdown("---")

    page = st.sidebar.radio(
        "Navigate",
        [
            "📊 Executive Summary",
            "🗺️ Property Map & Valuations",
            "📈 Market Trends & Forecasts",
            "🎯 Market Segmentation",
            "⚠️ Risk & Distress Analysis",
            "🎲 Monte Carlo Simulation",
            "🔗 Network Analysis",
        ],
    )

    st.sidebar.markdown("---")

    # Filters
    st.sidebar.markdown("### Filters")
    data = load_data()
    prop_df = data.get("property_valuations", pd.DataFrame())

    borough_filter = st.sidebar.multiselect(
        "Borough",
        options=sorted(prop_df["borough_name"].unique()) if "borough_name" in prop_df.columns else [],
        default=[],
    )

    price_range = st.sidebar.slider(
        "Price Range ($M)",
        min_value=0.0,
        max_value=100.0,
        value=(0.0, 100.0),
        step=1.0,
    )

    return page, borough_filter, price_range


# ─────────────────────────────────────────────────────────
# Page: Executive Summary
# ─────────────────────────────────────────────────────────
def page_executive_summary(data, filters):
    st.markdown('<p class="main-header">NYC Commercial Real Estate Investment Intelligence</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Data-driven insights for institutional CRE investment in New York City</p>', unsafe_allow_html=True)
    st.markdown("---")

    prop_df = apply_filters(data.get("property_valuations", pd.DataFrame()), filters)

    if prop_df.empty:
        st.warning("No data available. Run the pipeline first: `python run_pipeline.py`")
        return

    # KPI Metrics
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        total_value = prop_df["sale_price"].sum() / 1e9
        st.metric("Total Transaction Volume", f"${total_value:.1f}B")
    with col2:
        median_ppsf = prop_df["price_per_sqft"].median()
        st.metric("Median Price/SF", f"${median_ppsf:,.0f}")
    with col3:
        n_transactions = len(prop_df)
        st.metric("Transactions", f"{n_transactions:,}")
    with col4:
        if "distress_probability" in prop_df.columns:
            avg_distress = prop_df["distress_probability"].mean() * 100
            st.metric("Avg Distress Score", f"{avg_distress:.1f}%")
        else:
            st.metric("Avg Building Age", f"{prop_df['building_age'].mean():.0f}yr")
    with col5:
        if "sale_year" in prop_df.columns:
            yoy = prop_df.groupby("sale_year")["price_per_sqft"].median()
            if len(yoy) >= 2:
                change = (yoy.iloc[-1] / yoy.iloc[-2] - 1) * 100
                st.metric("YoY Price Change", f"{change:+.1f}%")

    st.markdown("---")

    # Charts row
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Price Distribution by Borough")
        fig = px.box(
            prop_df,
            x="borough_name",
            y="price_per_sqft",
            color="borough_name",
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig.update_layout(
            showlegend=False,
            yaxis_title="Price per SF ($)",
            xaxis_title="",
            height=400,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Transaction Volume Over Time")
        if "sale_year" in prop_df.columns:
            vol = prop_df.groupby("sale_year").agg(
                count=("sale_price", "count"),
                total_value=("sale_price", "sum"),
            ).reset_index()
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(
                go.Bar(x=vol["sale_year"], y=vol["count"], name="# Transactions",
                       marker_color="#667eea", opacity=0.7),
                secondary_y=False,
            )
            fig.add_trace(
                go.Scatter(x=vol["sale_year"], y=vol["total_value"] / 1e6,
                          name="Volume ($M)", line=dict(color="#e74c3c", width=3)),
                secondary_y=True,
            )
            fig.update_layout(height=400)
            fig.update_yaxes(title_text="Transactions", secondary_y=False)
            fig.update_yaxes(title_text="Volume ($M)", secondary_y=True)
            st.plotly_chart(fig, use_container_width=True)

    # Borough breakdown table
    st.subheader("Borough Market Summary")
    if "borough_name" in prop_df.columns:
        summary = prop_df.groupby("borough_name").agg({
            "sale_price": ["count", "mean", "median"],
            "price_per_sqft": ["mean", "median"],
            "building_age": "mean",
        }).round(0)
        summary.columns = [
            "# Sales", "Avg Price", "Median Price",
            "Avg $/SF", "Median $/SF", "Avg Age",
        ]
        summary = summary.sort_values("Median Price", ascending=False)

        # Format as currency
        for col in ["Avg Price", "Median Price"]:
            summary[col] = summary[col].apply(lambda x: f"${x:,.0f}")
        for col in ["Avg $/SF", "Median $/SF"]:
            summary[col] = summary[col].apply(lambda x: f"${x:,.0f}")

        st.dataframe(summary, use_container_width=True)


# ─────────────────────────────────────────────────────────
# Page: Property Map
# ─────────────────────────────────────────────────────────
def page_property_map(data, filters):
    st.header("🗺️ Property Map & Valuations")

    prop_df = apply_filters(data.get("property_valuations", pd.DataFrame()), filters)

    if prop_df.empty or "latitude" not in prop_df.columns:
        st.warning("No geospatial data available.")
        return

    # Color by selector
    color_by = st.selectbox(
        "Color properties by:",
        ["price_per_sqft", "risk_tier", "borough_name", "cluster", "building_age"],
    )

    # Sample for performance
    map_df = prop_df.dropna(subset=["latitude", "longitude"]).head(2000)

    # Folium map
    m = folium.Map(location=list(NYC_CENTER), zoom_start=11, tiles="CartoDB positron")

    # Color mapping
    color_map = {
        "Low Risk": "green", "Moderate Risk": "orange",
        "Elevated Risk": "red", "High Risk": "darkred",
        "Manhattan": "#e74c3c", "Brooklyn": "#3498db",
        "Queens": "#2ecc71", "Bronx": "#f39c12", "Staten Island": "#9b59b6",
    }

    for _, row in map_df.iterrows():
        lat, lon = row.get("latitude"), row.get("longitude")
        if pd.isna(lat) or pd.isna(lon):
            continue

        # Popup info
        popup_html = f"""
        <b>Price:</b> ${row.get('sale_price', 0):,.0f}<br>
        <b>$/SF:</b> ${row.get('price_per_sqft', 0):,.0f}<br>
        <b>Borough:</b> {row.get('borough_name', 'N/A')}<br>
        <b>Neighborhood:</b> {row.get('neighborhood', 'N/A')}<br>
        <b>Year Built:</b> {row.get('year_built', 'N/A')}<br>
        """

        if color_by in ["risk_tier", "borough_name"]:
            color = color_map.get(str(row.get(color_by, "")), "blue")
        else:
            val = float(row.get(color_by, 0) or 0)
            if val > prop_df[color_by].quantile(0.75):
                color = "red"
            elif val > prop_df[color_by].quantile(0.5):
                color = "orange"
            else:
                color = "green"

        folium.CircleMarker(
            location=[float(lat), float(lon)],
            radius=4,
            color=color,
            fill=True,
            fill_opacity=0.6,
            popup=folium.Popup(popup_html, max_width=250),
        ).add_to(m)

    st_folium(m, width=None, height=600)

    # Scatter plot: Price vs Size
    st.subheader("Price vs. Property Size")
    fig = px.scatter(
        prop_df.head(3000),
        x="gross_square_feet",
        y="sale_price",
        color="borough_name",
        size="num_floors",
        hover_data=["neighborhood", "price_per_sqft"],
        log_x=True,
        log_y=True,
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig.update_layout(height=500, xaxis_title="Gross Square Feet (log)", yaxis_title="Sale Price (log)")
    st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────────────────
# Page: Market Trends
# ─────────────────────────────────────────────────────────
def page_market_trends(data, filters):
    st.header("📈 Market Trends & Forecasts")

    prop_df = apply_filters(data.get("property_valuations", pd.DataFrame()), filters)
    macro_df = data.get("macro_economic", pd.DataFrame())

    if prop_df.empty:
        st.warning("No data available.")
        return

    # Time series by borough
    st.subheader("Median Price/SF Trends by Borough")
    if "sale_date" in prop_df.columns and "borough_name" in prop_df.columns:
        prop_df["sale_date"] = pd.to_datetime(prop_df["sale_date"], errors="coerce")
        prop_df["sale_month"] = prop_df["sale_date"].dt.to_period("Q").dt.to_timestamp()

        trends = prop_df.groupby(["sale_month", "borough_name"])["price_per_sqft"].median().reset_index()

        fig = px.line(
            trends,
            x="sale_month",
            y="price_per_sqft",
            color="borough_name",
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig.update_layout(
            height=450,
            xaxis_title="Quarter",
            yaxis_title="Median Price per SF ($)",
            legend_title="Borough",
        )
        st.plotly_chart(fig, use_container_width=True)

    # Macroeconomic indicators
    if not macro_df.empty:
        st.subheader("Macroeconomic Environment")
        st.markdown("*Interest rates, inflation, and employment shape CRE investment returns.*")

        col1, col2 = st.columns(2)

        with col1:
            if "fed_funds_rate" in macro_df.columns:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=macro_df.index if hasattr(macro_df.index, 'year') else macro_df.get("date", range(len(macro_df))),
                    y=macro_df["fed_funds_rate"],
                    fill="tozeroy",
                    fillcolor="rgba(102, 126, 234, 0.2)",
                    line=dict(color="#667eea", width=2),
                ))
                fig.update_layout(title="Federal Funds Rate (%)", height=350)
                st.plotly_chart(fig, use_container_width=True)

        with col2:
            if "unemployment_nyc" in macro_df.columns:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=macro_df.index if hasattr(macro_df.index, 'year') else macro_df.get("date", range(len(macro_df))),
                    y=macro_df["unemployment_nyc"],
                    fill="tozeroy",
                    fillcolor="rgba(231, 76, 60, 0.2)",
                    line=dict(color="#e74c3c", width=2),
                ))
                fig.update_layout(title="NYC Unemployment Rate (%)", height=350)
                st.plotly_chart(fig, use_container_width=True)

    # Year-over-year changes
    st.subheader("Annual Market Metrics")
    if "sale_year" in prop_df.columns:
        annual = prop_df.groupby("sale_year").agg({
            "sale_price": "median",
            "price_per_sqft": "median",
            "building_age": "mean",
        }).reset_index()

        annual["price_yoy"] = annual["price_per_sqft"].pct_change() * 100

        fig = make_subplots(rows=1, cols=2, subplot_titles=["Median Sale Price", "YoY Price Change (%)"])
        fig.add_trace(
            go.Bar(x=annual["sale_year"], y=annual["sale_price"] / 1e6,
                   marker_color="#667eea", name="Median Price ($M)"),
            row=1, col=1,
        )
        fig.add_trace(
            go.Bar(x=annual["sale_year"], y=annual["price_yoy"],
                   marker_color=["#2ecc71" if x >= 0 else "#e74c3c" for x in annual["price_yoy"].fillna(0)],
                   name="YoY Change (%)"),
            row=1, col=2,
        )
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────────────────
# Page: Market Segmentation
# ─────────────────────────────────────────────────────────
def page_segmentation(data, filters):
    st.header("🎯 Market Segmentation (Clustering Analysis)")
    st.markdown("""
    Properties and neighborhoods are grouped into distinct market segments
    using K-Means clustering on property characteristics. Each segment
    represents a unique risk/return profile.
    """)

    prop_df = apply_filters(data.get("property_valuations", pd.DataFrame()), filters)

    if prop_df.empty or "cluster" not in prop_df.columns:
        st.warning("No clustering data available. Run the pipeline first.")
        return

    # Cluster distribution
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Cluster Distribution")
        cluster_counts = prop_df["cluster"].value_counts().sort_index()
        fig = px.pie(
            values=cluster_counts.values,
            names=[f"Segment {i}" for i in cluster_counts.index],
            color_discrete_sequence=px.colors.qualitative.Set3,
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Cluster Profiles")
        profiles = prop_df.groupby("cluster").agg({
            "price_per_sqft": "median",
            "gross_square_feet": "median",
            "building_age": "median",
            "dist_to_cbd_km": "median",
            "sale_price": "count",
        }).round(1)
        profiles.columns = ["Median $/SF", "Median SF", "Median Age", "Dist to CBD (km)", "Count"]
        profiles.index = [f"Segment {i}" for i in profiles.index]
        st.dataframe(profiles, use_container_width=True)

    # 2D scatter of clusters
    st.subheader("Cluster Visualization: Price vs. Distance to CBD")
    fig = px.scatter(
        prop_df.head(3000),
        x="dist_to_cbd_km",
        y="price_per_sqft",
        color="cluster",
        color_continuous_scale="Viridis",
        hover_data=["borough_name", "neighborhood"],
        opacity=0.6,
    )
    fig.update_layout(
        height=500,
        xaxis_title="Distance to CBD (km)",
        yaxis_title="Price per SF ($)",
    )
    st.plotly_chart(fig, use_container_width=True)

    # Parallel coordinates
    st.subheader("Segment Comparison (Parallel Coordinates)")
    numeric_cols = ["price_per_sqft", "gross_square_feet", "building_age", "dist_to_cbd_km", "floor_area_ratio"]
    available_cols = [c for c in numeric_cols if c in prop_df.columns]
    if available_cols:
        sample = prop_df[available_cols + ["cluster"]].dropna().head(2000)
        fig = px.parallel_coordinates(
            sample,
            color="cluster",
            dimensions=available_cols,
            color_continuous_scale="Turbo",
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────────────────
# Page: Risk & Distress
# ─────────────────────────────────────────────────────────
def page_risk_analysis(data, filters):
    st.header("⚠️ Risk & Distress Analysis")
    st.markdown("""
    Properties are scored for distress probability using XGBoost classification
    based on building age, violation history, market position, and location factors.
    """)

    prop_df = apply_filters(data.get("property_valuations", pd.DataFrame()), filters)

    if prop_df.empty:
        st.warning("No data available.")
        return

    if "distress_probability" not in prop_df.columns:
        st.info("Distress model not yet run. Showing building risk indicators instead.")
        return

    # Risk tier distribution
    col1, col2, col3 = st.columns(3)

    with col1:
        high_risk = (prop_df["risk_tier"] == "High Risk").sum()
        st.metric("🔴 High Risk Properties", f"{high_risk:,}")
    with col2:
        elevated = (prop_df["risk_tier"] == "Elevated Risk").sum()
        st.metric("🟠 Elevated Risk", f"{elevated:,}")
    with col3:
        low_risk = (prop_df["risk_tier"] == "Low Risk").sum()
        st.metric("🟢 Low Risk", f"{low_risk:,}")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Risk Distribution")
        fig = px.histogram(
            prop_df,
            x="distress_probability",
            nbins=50,
            color="borough_name",
            marginal="box",
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig.update_layout(
            height=400,
            xaxis_title="Distress Probability",
            yaxis_title="Count",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Risk by Borough")
        risk_by_borough = prop_df.groupby("borough_name")["distress_probability"].mean().sort_values(ascending=False)
        fig = px.bar(
            x=risk_by_borough.index,
            y=risk_by_borough.values * 100,
            color=risk_by_borough.values,
            color_continuous_scale="RdYlGn_r",
        )
        fig.update_layout(
            height=400,
            xaxis_title="Borough",
            yaxis_title="Avg Distress Probability (%)",
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

    # Risk heatmap by neighborhood
    st.subheader("Risk Heatmap by Neighborhood")
    if "neighborhood" in prop_df.columns:
        risk_by_hood = prop_df.groupby("neighborhood").agg({
            "distress_probability": "mean",
            "sale_price": "count",
        }).reset_index()
        risk_by_hood.columns = ["Neighborhood", "Avg Distress Prob", "N Properties"]
        risk_by_hood = risk_by_hood[risk_by_hood["N Properties"] >= 10].sort_values("Avg Distress Prob", ascending=False)

        fig = px.bar(
            risk_by_hood.head(20),
            x="Neighborhood",
            y="Avg Distress Prob",
            color="Avg Distress Prob",
            color_continuous_scale="RdYlGn_r",
            text="N Properties",
        )
        fig.update_layout(height=450, xaxis_title="", yaxis_title="Average Distress Probability")
        st.plotly_chart(fig, use_container_width=True)

    # High-risk property list
    st.subheader("Top High-Risk Properties")
    high_risk_df = prop_df.nlargest(20, "distress_probability")[
        ["neighborhood", "borough_name", "sale_price", "price_per_sqft",
         "building_age", "distress_probability", "risk_tier"]
    ].copy()
    high_risk_df["sale_price"] = high_risk_df["sale_price"].apply(lambda x: f"${x:,.0f}")
    high_risk_df["distress_probability"] = high_risk_df["distress_probability"].apply(lambda x: f"{x:.1%}")
    st.dataframe(high_risk_df, use_container_width=True)


# ─────────────────────────────────────────────────────────
# Page: Monte Carlo Simulation
# ─────────────────────────────────────────────────────────
def page_simulation(data, filters):
    st.header("🎲 Monte Carlo Simulation — Investment Outcome Analysis")
    st.markdown("""
    Evaluate how investment outcomes may vary under different macroeconomic
    scenarios using Geometric Brownian Motion simulation.
    """)

    # User inputs
    col1, col2, col3 = st.columns(3)
    with col1:
        initial_value = st.number_input("Property Value ($)", value=10_000_000, step=1_000_000, format="%d")
    with col2:
        annual_drift = st.slider("Expected Annual Return (%)", -5.0, 15.0, 3.0) / 100
    with col3:
        annual_vol = st.slider("Annual Volatility (%)", 5.0, 30.0, 12.0) / 100

    col1, col2 = st.columns(2)
    with col1:
        n_sims = st.selectbox("Number of Simulations", [1000, 5000, 10000], index=1)
    with col2:
        horizon = st.selectbox("Investment Horizon (years)", [3, 5, 7, 10], index=1)

    if st.button("Run Simulation", type="primary"):
        np.random.seed(42)
        n_steps = horizon * 12
        dt = 1 / 12

        Z = np.random.standard_normal((n_sims, n_steps))
        paths = np.zeros((n_sims, n_steps + 1))
        paths[:, 0] = initial_value

        for t in range(n_steps):
            paths[:, t + 1] = paths[:, t] * np.exp(
                (annual_drift - 0.5 * annual_vol**2) * dt
                + annual_vol * np.sqrt(dt) * Z[:, t]
            )

        time_axis = np.arange(n_steps + 1) / 12
        terminal = paths[:, -1]
        returns = (terminal - initial_value) / initial_value

        # Results
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Expected Value", f"${terminal.mean():,.0f}")
        with col2:
            st.metric("Expected Return", f"{returns.mean():.1%}")
        with col3:
            var_95 = initial_value - np.percentile(terminal, 5)
            st.metric("VaR (95%)", f"${var_95:,.0f}")
        with col4:
            prob_loss = (terminal < initial_value).mean()
            st.metric("P(Loss)", f"{prob_loss:.1%}")

        # Fan chart
        percentiles = [5, 25, 50, 75, 95]
        pct_values = {p: np.percentile(paths, p, axis=0) for p in percentiles}

        fig = go.Figure()

        # Confidence bands
        fig.add_trace(go.Scatter(
            x=np.concatenate([time_axis, time_axis[::-1]]),
            y=np.concatenate([pct_values[95], pct_values[5][::-1]]),
            fill="toself", fillcolor="rgba(102, 126, 234, 0.1)",
            line=dict(color="rgba(255,255,255,0)"),
            name="90% CI",
        ))
        fig.add_trace(go.Scatter(
            x=np.concatenate([time_axis, time_axis[::-1]]),
            y=np.concatenate([pct_values[75], pct_values[25][::-1]]),
            fill="toself", fillcolor="rgba(102, 126, 234, 0.2)",
            line=dict(color="rgba(255,255,255,0)"),
            name="50% CI",
        ))
        fig.add_trace(go.Scatter(
            x=time_axis, y=pct_values[50],
            line=dict(color="#667eea", width=3),
            name="Median",
        ))
        fig.add_hline(y=initial_value, line_dash="dash", line_color="red", annotation_text="Initial Value")

        fig.update_layout(
            title=f"Property Value Projection — {n_sims:,} Simulations",
            xaxis_title="Years",
            yaxis_title="Property Value ($)",
            height=500,
        )
        st.plotly_chart(fig, use_container_width=True)

        # Terminal value distribution
        col1, col2 = st.columns(2)
        with col1:
            fig = px.histogram(
                x=terminal / 1e6, nbins=100,
                labels={"x": "Terminal Value ($M)"},
                color_discrete_sequence=["#667eea"],
            )
            fig.add_vline(x=initial_value / 1e6, line_dash="dash", line_color="red")
            fig.update_layout(title="Terminal Value Distribution", height=400)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = px.histogram(
                x=returns * 100, nbins=100,
                labels={"x": "Total Return (%)"},
                color_discrete_sequence=["#2ecc71"],
            )
            fig.add_vline(x=0, line_dash="dash", line_color="red")
            fig.update_layout(title="Return Distribution", height=400)
            st.plotly_chart(fig, use_container_width=True)

        # Scenario comparison
        st.subheader("Scenario Analysis")
        scenarios = {
            "Base Case": (annual_drift, annual_vol),
            "Bull Case": (annual_drift + 0.02, annual_vol * 0.8),
            "Bear Case": (annual_drift - 0.03, annual_vol * 1.3),
            "Stress Test": (-0.08, annual_vol * 2.0),
        }

        scenario_data = []
        for name, (d, v) in scenarios.items():
            Z_s = np.random.standard_normal((1000, n_steps))
            p = np.zeros((1000, n_steps + 1))
            p[:, 0] = initial_value
            for t in range(n_steps):
                p[:, t + 1] = p[:, t] * np.exp((d - 0.5 * v**2) * dt + v * np.sqrt(dt) * Z_s[:, t])
            term = p[:, -1]
            scenario_data.append({
                "Scenario": name,
                "Expected Value ($M)": f"{term.mean() / 1e6:.1f}",
                "Expected Return": f"{((term.mean() - initial_value) / initial_value):.1%}",
                "VaR 95% ($M)": f"{(initial_value - np.percentile(term, 5)) / 1e6:.1f}",
                "P(Loss)": f"{(term < initial_value).mean():.1%}",
            })

        st.dataframe(pd.DataFrame(scenario_data), use_container_width=True)


# ─────────────────────────────────────────────────────────
# Page: Network Analysis
# ─────────────────────────────────────────────────────────
def page_network(data, filters):
    st.header("🔗 Ownership Network Analysis")
    st.markdown("""
    Graph-based analysis of property ownership networks reveals hidden connections
    between market participants, portfolio concentrations, and potential deal flow.
    *In production, this leverages Neo4j graph database for real-time querying.*
    """)

    prop_df = apply_filters(data.get("property_valuations", pd.DataFrame()), filters)

    if prop_df.empty:
        st.warning("No data available.")
        return

    # Simulated network metrics for demonstration
    st.subheader("Market Participant Concentration")

    if "borough_name" in prop_df.columns and "neighborhood" in prop_df.columns:
        # Neighborhood connectivity
        hood_counts = prop_df.groupby(["borough_name", "neighborhood"]).size().reset_index(name="transactions")

        fig = px.treemap(
            hood_counts,
            path=["borough_name", "neighborhood"],
            values="transactions",
            color="transactions",
            color_continuous_scale="Blues",
        )
        fig.update_layout(title="Market Activity by Borough & Neighborhood", height=500)
        st.plotly_chart(fig, use_container_width=True)

    # Portfolio concentration (Herfindahl)
    st.subheader("Neo4j Graph Database Integration")
    st.markdown("""
    The production system uses Neo4j to model:
    - **Owner → Property** relationships (ownership graph)
    - **Owner ↔ Owner** co-investment networks
    - **Geographic proximity** clusters
    - **Transaction flow** (buyer → seller chains)

    Sample Cypher queries are provided in `src/graph_analysis.py`.
    """)

    st.code("""
    // Find owners with the largest portfolios
    MATCH (o:Owner)-[:OWNS]->(p:Property)
    WITH o, COUNT(p) AS n_properties,
         SUM(p.assessed_value) AS portfolio_value
    ORDER BY portfolio_value DESC LIMIT 20
    RETURN o.name AS owner, n_properties, portfolio_value
    """, language="cypher")

    st.code("""
    // Find co-investment networks (owners in same neighborhoods)
    MATCH (o1:Owner)-[:OWNS]->(p1:Property)-[:LOCATED_IN]->(n:Neighborhood)
          <-[:LOCATED_IN]-(p2:Property)<-[:OWNS]-(o2:Owner)
    WHERE o1 <> o2
    WITH o1, o2, COUNT(DISTINCT n) AS shared_neighborhoods
    WHERE shared_neighborhoods >= 3
    RETURN o1.name, o2.name, shared_neighborhoods
    ORDER BY shared_neighborhoods DESC LIMIT 50
    """, language="cypher")


# ─────────────────────────────────────────────────────────
# Utility Functions
# ─────────────────────────────────────────────────────────
def apply_filters(df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    """Apply sidebar filters to DataFrame."""
    if df.empty:
        return df

    borough_filter = filters.get("borough", [])
    price_range = filters.get("price_range", (0, 100))

    if borough_filter and "borough_name" in df.columns:
        df = df[df["borough_name"].isin(borough_filter)]

    if "sale_price" in df.columns:
        df = df[
            (df["sale_price"] >= price_range[0] * 1e6)
            & (df["sale_price"] <= price_range[1] * 1e6)
        ]

    return df


# ─────────────────────────────────────────────────────────
# Main App
# ─────────────────────────────────────────────────────────
def main():
    page, borough_filter, price_range = sidebar()
    data = load_data()
    filters = {"borough": borough_filter, "price_range": price_range}

    if page == "📊 Executive Summary":
        page_executive_summary(data, filters)
    elif page == "🗺️ Property Map & Valuations":
        page_property_map(data, filters)
    elif page == "📈 Market Trends & Forecasts":
        page_market_trends(data, filters)
    elif page == "🎯 Market Segmentation":
        page_segmentation(data, filters)
    elif page == "⚠️ Risk & Distress Analysis":
        page_risk_analysis(data, filters)
    elif page == "🎲 Monte Carlo Simulation":
        page_simulation(data, filters)
    elif page == "🔗 Network Analysis":
        page_network(data, filters)


if __name__ == "__main__":
    main()
