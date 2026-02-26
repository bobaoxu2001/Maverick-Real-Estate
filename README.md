# NYC Commercial Real Estate Investment Intelligence

**Data-driven analytics platform for institutional CRE investment in New York City**

A comprehensive machine learning and data engineering project that acquires, processes, and models NYC commercial real estate data to generate actionable investment intelligence — including property valuation, market segmentation, distress prediction, time series forecasting, and Monte Carlo risk simulation.

---

## Overview

This project demonstrates a production-grade data science pipeline for commercial real estate (CRE) investment analysis, designed to surface actionable insights from NYC's rich municipal and commercial data ecosystem.

### Key Capabilities

| Capability | Description | Technique |
|---|---|---|
| **Property Valuation** | Hedonic regression decomposing value into constituent features | OLS, Ridge, Gradient Boosting, Stacked Ensemble |
| **Market Segmentation** | Unsupervised clustering of properties and neighborhoods | K-Means, DBSCAN, PCA |
| **Price Forecasting** | Time series modeling of market trends with macro variables | SARIMAX, Prophet |
| **Distress Prediction** | Probability of financial distress / forced sale | XGBoost with calibrated probabilities |
| **Investment Simulation** | Monte Carlo scenario analysis for risk management | Geometric Brownian Motion |
| **Network Analysis** | Ownership graph revealing hidden market connections | NetworkX + Neo4j Cypher |

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                        DATA SOURCES (APIs)                          │
│  ┌────────────────┐  ┌──────────────┐  ┌──────────────────────────┐ │
│  │ NYC Open Data   │  │  FRED API    │  │  Neo4j Graph Database    │ │
│  │ (Socrata API)   │  │              │  │  (Ownership Network)     │ │
│  │ • PLUTO         │  │ • Fed Funds  │  │  • Owner→Property        │ │
│  │ • DOF Sales     │  │ • 10Y Treas  │  │  • Co-investment         │ │
│  │ • DOB Permits   │  │ • CPI        │  │  • Transaction Flow      │ │
│  │ • HPD Violations│  │ • NYC Unemp  │  │                          │ │
│  └───────┬────────┘  └──────┬───────┘  └────────────┬─────────────┘ │
└──────────┼──────────────────┼───────────────────────┼───────────────┘
           │                  │                       │
           ▼                  ▼                       ▼
┌──────────────────────────────────────────────────────────────────────┐
│                     ELT PIPELINE (dbt + Prefect)                    │
│  ┌─────────────┐   ┌───────────────┐   ┌──────────────────────────┐ │
│  │  STAGING     │──▶│ INTERMEDIATE  │──▶│       MARTS              │ │
│  │ stg_pluto    │   │ int_property  │   │ mart_property_valuations │ │
│  │ stg_sales    │   │ _features     │   │ mart_market_segments     │ │
│  │ stg_permits  │   │               │   │                          │ │
│  └─────────────┘   └───────────────┘   └──────────────────────────┘ │
└──────────────────────────────────────────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────────────────────────────────┐
│                    FEATURE ENGINEERING                               │
│  • Geospatial (haversine distances to CBD, landmarks)               │
│  • Property (FAR, building age, renovation status, log transforms)  │
│  • Market Context (neighborhood stats, permit activity, violations) │
│  • Categorical Encoding (one-hot, label, target encoding)           │
│  • Dimensionality Reduction (PCA with variance thresholding)        │
└──────────────────────────────────────────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────────────────────────────────┐
│                     PREDICTIVE MODELS                                │
│  ┌───────────────┐ ┌──────────────┐ ┌───────────────┐ ┌──────────┐ │
│  │   HEDONIC      │ │  CLUSTERING  │ │ TIME SERIES   │ │ DISTRESS │ │
│  │  REGRESSION    │ │              │ │               │ │ PREDICTOR│ │
│  │ • OLS          │ │ • K-Means    │ │ • SARIMAX     │ │ • XGBoost│ │
│  │ • Ridge/Lasso  │ │ • DBSCAN     │ │ • Prophet     │ │ • Calib. │ │
│  │ • Grad Boost   │ │ • PCA viz    │ │ • Momentum    │ │ • ROC/PR │ │
│  └───────────────┘ └──────────────┘ └───────────────┘ └──────────┘ │
│  ┌───────────────────────────────┐ ┌───────────────────────────────┐ │
│  │  MONTE CARLO SIMULATION       │ │  GRAPH / NETWORK ANALYSIS     │ │
│  │ • GBM value paths             │ │ • Ownership bipartite graph   │ │
│  │ • Scenario analysis           │ │ • Community detection         │ │
│  │ • Portfolio simulation        │ │ • Propensity-to-sell signals  │ │
│  │ • VaR / CVaR estimation       │ │ • Neo4j Cypher production     │ │
│  └───────────────────────────────┘ └───────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────────────────────────────────┐
│                   PRESENTATION LAYER                                 │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │  Streamlit Dashboard (interactive, BI-tool-ready)              │  │
│  │  • Executive Summary KPIs    • Risk Heatmaps                  │  │
│  │  • Folium Property Map       • Monte Carlo Fan Charts         │  │
│  │  • Time Series Trends        • Network Visualization          │  │
│  │  • Cluster Scatter Plots     • Scenario Comparison Tables     │  │
│  └────────────────────────────────────────────────────────────────┘  │
│  Compatible with: Looker, BigQuery ML, Google Cloud Platform        │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
nyc-cre-investment-analytics/
├── README.md
├── requirements.txt
├── .gitignore
├── .env.example
├── config.py                          # Centralized configuration
├── run_pipeline.py                    # Full pipeline orchestrator
│
├── src/
│   ├── data_acquisition.py            # NYC Open Data + FRED API clients
│   ├── elt_pipeline.py                # ELT: staging → intermediate → marts
│   ├── feature_engineering.py         # Geospatial, property, market features
│   ├── graph_analysis.py              # Neo4j-compatible network analysis
│   ├── reporting/
│   │   ├── __init__.py
│   │   └── model_reports.py           # HTML + PNG performance report generator
│   └── models/
│       ├── hedonic_regression.py      # Property valuation (OLS/Ridge/GBM)
│       ├── property_clustering.py     # Market segmentation (K-Means/DBSCAN)
│       ├── time_series_forecast.py    # SARIMAX + Prophet forecasting
│       ├── distress_predictor.py      # Default probability (XGBoost)
│       └── simulation.py             # Monte Carlo investment simulation
│
├── dbt_models/                        # Production-ready dbt SQL models
│   ├── dbt_project.yml
│   └── models/
│       ├── staging/                   # Raw → clean (stg_pluto, stg_sales, ...)
│       ├── intermediate/              # Feature enrichment joins
│       └── marts/                     # Final analytical tables
│
├── app/
│   └── dashboard.py                   # Streamlit interactive dashboard
│
├── docs/
│   └── data_dictionary.md             # Data schema + business glossary
│
├── data/
│   ├── raw/                           # API-fetched data (parquet)
│   └── processed/                     # Pipeline output (parquet)
│
├── models/                            # Saved model artifacts (joblib)
└── reports/                           # Logs + model performance reports
```

---

## Data Sources

All data is sourced from **real, publicly available APIs** — no synthetic or sample data required.

| Source | Dataset | API | Records | Use Case |
|---|---|---|---|---|
| NYC Open Data | **PLUTO** | Socrata | ~870K lots | Property characteristics, zoning, assessed values |
| NYC Open Data | **DOF Rolling Sales** | Socrata | ~100K/yr | Actual transaction prices |
| NYC Open Data | **DOB Permits** | Socrata | ~500K | Development activity signals |
| NYC Open Data | **HPD Violations** | Socrata | ~1M | Distress / deferred maintenance signals |
| Federal Reserve | **FRED** | REST API | 15yr monthly | Fed Funds, 10Y Treasury, CPI, unemployment |

---

## Models & Methodology

### 1. Hedonic Regression — Property Valuation

The hedonic pricing model decomposes property value into the implicit prices of individual characteristics:

**log(Price) = β₀ + β₁·Size + β₂·Age + β₃·Location + β₄·Amenities + ε**

- **OLS with HC3 robust standard errors** — interpretable coefficients for investment memos
- **Ridge/Lasso regression** — regularization for high-dimensional feature sets
- **Gradient Boosted Regression** — captures non-linear price-feature relationships

*Business value: Identifies mispriced assets where hedonic fair value diverges from market price.*

### 2. Property Clustering — Market Segmentation

Unsupervised learning discovers natural property groupings:

- **K-Means** with silhouette-optimized cluster count — interpretable market segments
- **DBSCAN** — density-based spatial clustering for geographic sub-markets
- **PCA** for dimensionality reduction and 2D/3D visualization

*Business value: Identifies comparable properties (comps) and emerging sub-market segments.*

### 3. Time Series Forecasting — Market Trends

Models temporal dynamics in NYC CRE markets:

- **SARIMAX** — seasonal ARIMA with macroeconomic exogenous variables (interest rates, CPI)
- **Prophet** — decomposable forecasting with automatic changepoint detection (market regime shifts)
- **Rolling momentum indicators** — trend signals for investment timing

*Business value: Forecasts market conditions over investment horizons (3-10 years).*

### 4. Distress Prediction — Risk Management

Classifies property distress probability using observable signals:

- **XGBoost** with class weighting for imbalanced data
- **Platt scaling** for calibrated probability estimates
- Features: violations, building age, market position, location, renovation status

*Business value: Early warning system for portfolio risk and opportunistic acquisition targeting.*

### 5. Monte Carlo Simulation — Investment Outcomes

Stochastic modeling of investment outcomes under uncertainty:

- **Geometric Brownian Motion** for property value paths
- **Multi-scenario analysis**: Base, Bull, Bear, Stress Test (GFC-like)
- **Portfolio simulation** with correlated property returns
- **Risk metrics**: VaR, CVaR, probability of loss, Sharpe ratio

*Business value: Quantifies downside risk for investment committee decision-making.*

### 6. Network Analysis — Market Intelligence

Graph-based modeling of ownership networks:

- **Bipartite graph** (Owner ↔ Property) with NetworkX
- **Community detection** (Louvain algorithm) for market clusters
- **Propensity-to-sell scoring** from network signals
- **Production Neo4j Cypher queries** for scalable graph querying

*Business value: Surfaces deal flow opportunities and hidden market connections.*

---

## Quick Start

### Prerequisites

- Python 3.10+
- (Optional) FRED API key — [get free key](https://fred.stlouisfed.org/docs/api/api_key.html)
- (Optional) NYC Open Data app token — [get free token](https://data.cityofnewyork.us/profile/edit/developer_settings)

### Installation

```bash
# Clone the repository
git clone https://github.com/allenxu/nyc-cre-investment-analytics.git
cd nyc-cre-investment-analytics

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure API keys (optional but recommended)
cp .env.example .env
# Edit .env with your API keys
```

### Run the Pipeline

```bash
# Full pipeline with live API data
python run_pipeline.py

# Larger professional-scale pull (more rows from APIs)
python run_pipeline.py --data-scale professional

# Institutional-scale pull (largest preset)
python run_pipeline.py --data-scale institutional

# Demo mode (no API keys required — uses realistic generated data)
python run_pipeline.py --demo

# Run specific steps
python run_pipeline.py --step acquire    # Fetch data only
python run_pipeline.py --step model      # Train models only
python run_pipeline.py --step simulate   # Run simulations only

# Optional: customize raw pull limits directly
python run_pipeline.py --step acquire --limit-pluto 200000 --limit-sales 250000
```

### Launch Dashboard

```bash
streamlit run app/dashboard.py
```

The dashboard opens at `http://localhost:8501` with interactive visualizations:
- Executive summary with KPIs
- Interactive property map (Folium)
- Market trend charts with macro overlay
- Cluster visualization with parallel coordinates
- Risk heatmaps and distress scoring
- Monte Carlo fan charts and scenario tables
- Network analysis and Neo4j integration preview
- Model diagnostics gallery from generated report figures

---

## Professional Enhancements (Maverick-style)

- **More data**: configurable acquisition profiles (`quick`, `professional`, `institutional`) with higher default limits and full pagination.
- **Better models**: hedonic stack ensemble (Ridge + Gradient Boosting + Random Forest) for stronger predictive performance.
- **More visuals**: automated reporting pipeline outputs model leaderboard, feature-importance charts, and market diagnostics images.
- **Professional artifacts**: timestamped HTML report bundles in `reports/model_performance/` and project data dictionary in `docs/data_dictionary.md`.

---

## Technology Stack

| Category | Technologies |
|---|---|
| **Languages** | Python, SQL |
| **Data Acquisition** | Socrata API (sodapy), FRED API (fredapi), REST APIs |
| **Data Engineering** | pandas, dbt (BigQuery), Prefect (orchestration) |
| **ML / Statistics** | scikit-learn, XGBoost, statsmodels, Prophet |
| **Graph Database** | Neo4j (Cypher), NetworkX |
| **Visualization** | Plotly, Folium (geospatial), Matplotlib, Seaborn |
| **Dashboard / BI** | Streamlit (Looker-compatible design patterns) |
| **Cloud** | Google Cloud (BigQuery, GCS) — production target |
| **Geospatial** | GeoPandas, Shapely, Folium |

---

## Key Design Decisions

1. **Log-linear hedonic specification**: Captures the multiplicative nature of property pricing (a 10% increase in sqft doesn't add a fixed dollar amount — it multiplies the price).

2. **dbt layered architecture** (staging → intermediate → marts): Ensures data lineage, testability, and separation of concerns — matching production patterns at data-forward CRE firms.

3. **Calibrated probability models** for distress prediction: Raw classifier scores are not true probabilities. Platt scaling ensures that a 30% distress score truly means ~30% probability, critical for risk-adjusted investment scoring.

4. **Monte Carlo with macro-correlated scenarios**: Investment outcomes depend on systematic (macro) factors, not just property-specific risk. Scenario analysis captures tail risk that point estimates miss.

5. **Neo4j-compatible graph analysis**: The ownership network is inherently a graph problem. Providing both local (NetworkX) and production (Neo4j Cypher) implementations demonstrates understanding of when graph databases add value.

---

## Future Enhancements

- [ ] NLP pipeline for lease abstractions and offering memorandums
- [ ] Computer vision for satellite imagery change detection
- [ ] Real-time data streaming with Apache Kafka
- [ ] MLflow experiment tracking and model registry
- [ ] Airflow/Prefect production orchestration DAGs
- [ ] API service layer (FastAPI) for model serving
- [ ] Integration with CoStar / REIS commercial data vendors

---

## Author

**Allen Xu**

Built as a demonstration of data science capabilities for commercial real estate investment analysis — combining domain expertise in NYC CRE markets with production-grade ML engineering.

---

*Data sourced from NYC Open Data and Federal Reserve Economic Data (FRED). All analysis is for demonstration purposes and does not constitute investment advice.*
