#!/usr/bin/env python3
"""
NYC CRE Investment Analytics — Full Pipeline Runner
=====================================================
Orchestrates the complete analytics pipeline:

  1. Data Acquisition — Fetch from NYC Open Data & FRED APIs
  2. ELT Pipeline — Clean, transform, and load (staging → intermediate → marts)
  3. Feature Engineering — Geospatial, property, market context features
  4. Model Training — Hedonic regression, clustering, time series, distress prediction
  5. Simulation — Monte Carlo scenario analysis
  6. Graph Analysis — Ownership network construction

Usage:
    python run_pipeline.py                    # Run full pipeline
    python run_pipeline.py --step acquire     # Run specific step
    python run_pipeline.py --step model       # Train models only
    python run_pipeline.py --demo             # Run with demo data (no API needed)

Author: Allen Xu
"""

import argparse
import time
from pathlib import Path
from loguru import logger
import pandas as pd
import sys

# Configure logging
logger.remove()
logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {level:<8} | {message}")
logger.add("reports/pipeline.log", level="DEBUG", rotation="10 MB")

# Project imports
sys.path.insert(0, str(Path(__file__).parent))
from config import (
    DATA_RAW,
    DATA_PROCESSED,
    MODEL_ARTIFACTS,
    DATA_FETCH_LIMITS,
    DATA_SCALE_PROFILES,
    DEFAULT_DATA_SCALE,
    DEMO_SAMPLE_SIZE,
)


def _resolve_fetch_limits(
    data_scale: str,
    limit_pluto: int | None = None,
    limit_sales: int | None = None,
    limit_permits: int | None = None,
    limit_violations: int | None = None,
) -> dict:
    """Resolve effective data limits from profile + optional overrides."""
    base_limits = DATA_SCALE_PROFILES.get(data_scale, DATA_FETCH_LIMITS).copy()

    overrides = {
        "pluto": limit_pluto,
        "rolling_sales": limit_sales,
        "dob_permits": limit_permits,
        "hpd_violations": limit_violations,
    }
    for key, value in overrides.items():
        if value is not None and value > 0:
            base_limits[key] = value
    return base_limits


def step_acquire(
    use_demo: bool = False,
    fetch_limits: dict | None = None,
    demo_size: int = DEMO_SAMPLE_SIZE,
):
    """Step 1: Data Acquisition from APIs."""
    logger.info("=" * 60)
    logger.info("STEP 1: DATA ACQUISITION")
    logger.info("=" * 60)

    if use_demo:
        logger.info(f"Using demo data (no API calls, sample size={demo_size:,})")
        from app.dashboard import generate_demo_data
        data = generate_demo_data(n=demo_size)
        for name, df in data.items():
            path = DATA_RAW / f"{name}.parquet"
            df.to_parquet(path, index=isinstance(df.index, pd.DatetimeIndex))
            logger.info(f"  Saved demo {name}: {df.shape}")
        return data

    from src.data_acquisition import NYCOpenDataClient, FREDClient, save_raw_data

    # NYC Open Data
    logger.info("Fetching NYC Open Data...")
    nyc_client = NYCOpenDataClient()
    fetch_limits = fetch_limits or DATA_FETCH_LIMITS
    logger.info(f"Using fetch limits: {fetch_limits}")
    nyc_data = nyc_client.fetch_all(limits=fetch_limits)

    # FRED Macroeconomic Data
    logger.info("Fetching FRED macroeconomic data...")
    fred_client = FREDClient()
    macro_df = fred_client.fetch_all_macro()
    nyc_data["macro_economic"] = macro_df

    # Save raw data
    save_raw_data(nyc_data)

    return nyc_data


def step_elt(raw_data: dict = None):
    """Step 2: ELT Pipeline (Staging → Intermediate → Marts)."""
    logger.info("=" * 60)
    logger.info("STEP 2: ELT PIPELINE")
    logger.info("=" * 60)

    from src.elt_pipeline import run_elt_pipeline
    from src.data_acquisition import load_raw_data

    if raw_data is None:
        raw_data = load_raw_data()

    results = run_elt_pipeline(raw_data)
    return results


def step_features(elt_data: dict = None):
    """Step 3: Feature Engineering."""
    logger.info("=" * 60)
    logger.info("STEP 3: FEATURE ENGINEERING")
    logger.info("=" * 60)

    from src.feature_engineering import engineer_features

    # Load data
    pluto = pd.DataFrame()
    sales = pd.DataFrame()
    permits = None
    violations = None

    if elt_data is None:
        prop_path = DATA_PROCESSED / "property_valuations.parquet"
        if prop_path.exists():
            sales = pd.read_parquet(prop_path)
        pluto_path = DATA_PROCESSED / "staging_pluto.parquet"
        if pluto_path.exists():
            pluto = pd.read_parquet(pluto_path)
        permits_path = DATA_PROCESSED / "staging_permits.parquet"
        if permits_path.exists():
            permits = pd.read_parquet(permits_path)
        violations_path = DATA_PROCESSED / "staging_violations.parquet"
        if violations_path.exists():
            violations = pd.read_parquet(violations_path)
    else:
        pluto = elt_data.get("staging", {}).get("pluto", pd.DataFrame())
        sales = elt_data.get("property_valuations", pd.DataFrame())
        permits = elt_data.get("staging", {}).get("permits", None)
        violations = elt_data.get("staging", {}).get("violations", None)

    if sales.empty:
        logger.warning("No sales data available for feature engineering")
        return pd.DataFrame()

    featured_df = engineer_features(pluto, sales, permits, violations)

    # Save
    featured_df.to_parquet(DATA_PROCESSED / "featured_properties.parquet", index=False)
    logger.info(f"Featured dataset: {featured_df.shape}")

    return featured_df


def step_models(
    featured_df: pd.DataFrame = None,
    macro_df: pd.DataFrame = None,
    violations_df: pd.DataFrame = None,
):
    """Step 4: Model Training."""
    logger.info("=" * 60)
    logger.info("STEP 4: MODEL TRAINING")
    logger.info("=" * 60)

    # Load data if not provided
    if featured_df is None:
        path = DATA_PROCESSED / "featured_properties.parquet"
        if path.exists():
            featured_df = pd.read_parquet(path)
        else:
            path = DATA_PROCESSED / "property_valuations.parquet"
            if path.exists():
                featured_df = pd.read_parquet(path)
            else:
                logger.error("No featured data available. Run feature engineering first.")
                return {}

    if macro_df is None:
        macro_path = DATA_PROCESSED / "macro_economic.parquet"
        if macro_path.exists():
            macro_df = pd.read_parquet(macro_path)

    if violations_df is None:
        violations_path = DATA_PROCESSED / "staging_violations.parquet"
        if violations_path.exists():
            violations_df = pd.read_parquet(violations_path)

    results = {}

    # ── 4a: Hedonic Regression ──
    logger.info("Training Hedonic Regression...")
    try:
        from src.models.hedonic_regression import HedonicRegressionModel
        hedonic = HedonicRegressionModel()

        if "sale_price" in featured_df.columns:
            # Filter for valid regression data
            reg_df = featured_df[
                featured_df["sale_price"].notna() &
                (featured_df["sale_price"] > 0)
            ].copy()

            if len(reg_df) > 100:
                # OLS for interpretability
                ols_results = hedonic.fit_ols(reg_df)
                results["hedonic_ols"] = ols_results

                # Ridge for prediction
                ridge_results = hedonic.fit_regularized(reg_df, model_type="ridge")
                results["hedonic_ridge"] = ridge_results

                # Gradient Boosting for accuracy
                gb_results = hedonic.fit_gradient_boosting(reg_df)
                results["hedonic_gb"] = gb_results

                # Stacked ensemble for stronger predictive performance
                stacked_results = hedonic.fit_stacked_ensemble(reg_df)
                results["hedonic_stacked"] = stacked_results

                hedonic.save()
            else:
                logger.warning(f"Insufficient data for regression ({len(reg_df)} rows)")
    except Exception as e:
        logger.error(f"Hedonic regression failed: {e}")

    # ── 4b: Clustering ──
    logger.info("Training Property Clustering...")
    try:
        from src.models.property_clustering import PropertyClusteringModel
        clustering = PropertyClusteringModel()

        # Find optimal k
        optimal_result = clustering.find_optimal_k(featured_df)
        results["optimal_k"] = optimal_result

        # Fit K-Means
        clustered_df = clustering.fit_kmeans(featured_df)
        results["clustering"] = clustering.results.get("kmeans", {})

        # Get cluster profiles
        profiles = clustering.get_cluster_profiles(clustered_df)
        results["cluster_profiles"] = profiles.to_dict()

        # Save clustered data
        clustered_df.to_parquet(DATA_PROCESSED / "clustered_properties.parquet", index=False)
        clustering.save()
    except Exception as e:
        logger.error(f"Clustering failed: {e}")

    # ── 4c: Time Series ──
    logger.info("Training Time Series Models...")
    try:
        from src.models.time_series_forecast import TimeSeriesForecaster
        ts = TimeSeriesForecaster()

        if "sale_date" in featured_df.columns and "sale_price" in featured_df.columns:
            price_series = ts.prepare_price_series(featured_df, metric="median")

            if isinstance(price_series, pd.DataFrame) and len(price_series) > 24:
                # Stationarity test
                if "sale_price" in price_series.columns:
                    stationarity = ts.test_stationarity(price_series["sale_price"])
                    results["stationarity"] = stationarity

                    # SARIMAX
                    sarimax_results = ts.fit_sarimax(
                        price_series["sale_price"],
                        exog=macro_df if macro_df is not None else None,
                    )
                    results["sarimax"] = sarimax_results

                    # Momentum indicators
                    momentum = ts.compute_momentum_indicators(price_series["sale_price"])
                    results["momentum"] = {"latest": momentum.iloc[-1].to_dict() if len(momentum) > 0 else {}}

                ts.save()
    except Exception as e:
        logger.error(f"Time series failed: {e}")

    # ── 4d: Distress Predictor ──
    logger.info("Training Distress Predictor...")
    try:
        from src.models.distress_predictor import DistressPredictor
        distress = DistressPredictor()

        # Create labels
        labeled_df = distress.create_distress_labels(featured_df, violations_df=violations_df)

        if labeled_df["is_distressed"].sum() > 20:
            distress_results = distress.fit(labeled_df)
            results["distress"] = {
                k: v for k, v in distress_results.items()
                if k not in ["roc_curve", "pr_curve"]  # Skip non-serializable
            }

            # Score all properties
            scored_df = distress.predict_distress_probability(labeled_df)
            scored_df.to_parquet(DATA_PROCESSED / "scored_properties.parquet", index=False)
            distress.save()
        else:
            logger.warning("Insufficient distress cases for modeling")
    except Exception as e:
        logger.error(f"Distress predictor failed: {e}")

    # ── 4e: Automated reporting (figures + html) ──
    try:
        from src.reporting.model_reports import ModelReportGenerator

        report_generator = ModelReportGenerator()
        report_artifacts = report_generator.generate(results, featured_df)
        results["model_report"] = report_artifacts
    except Exception as e:
        logger.error(f"Model report generation failed: {e}")

    return results


def step_simulation():
    """Step 5: Monte Carlo Simulation."""
    logger.info("=" * 60)
    logger.info("STEP 5: MONTE CARLO SIMULATION")
    logger.info("=" * 60)

    from src.models.simulation import MonteCarloSimulator
    sim = MonteCarloSimulator()

    # Single property simulation
    logger.info("Running single property simulation...")
    single_result = sim.simulate_property_value(
        initial_value=10_000_000,
        annual_drift=0.03,
        annual_volatility=0.12,
    )

    # Scenario analysis
    logger.info("Running scenario analysis...")
    scenario_results = sim.scenario_analysis(
        initial_value=10_000_000,
        base_drift=0.03,
        base_vol=0.12,
    )

    # Portfolio simulation
    logger.info("Running portfolio simulation...")
    portfolio_result = sim.portfolio_simulation(
        property_values=[10e6, 15e6, 8e6, 20e6],
        property_drifts=[0.03, 0.04, 0.02, 0.035],
        property_vols=[0.12, 0.15, 0.10, 0.13],
    )

    return {
        "single_property": {
            k: v for k, v in single_result.items()
            if k not in ["paths", "percentiles"]
        },
        "scenarios": {
            name: {k: v for k, v in s.items() if k != "percentiles"}
            for name, s in scenario_results.items()
        },
        "portfolio": {
            k: v for k, v in portfolio_result.items()
            if k not in ["property_paths", "portfolio_paths"]
        },
    }


def step_graph(featured_df: pd.DataFrame = None):
    """Step 6: Graph / Network Analysis."""
    logger.info("=" * 60)
    logger.info("STEP 6: GRAPH ANALYSIS")
    logger.info("=" * 60)

    from src.graph_analysis import CRENetworkAnalyzer

    if featured_df is None:
        for name in ["featured_properties", "property_valuations"]:
            path = DATA_PROCESSED / f"{name}.parquet"
            if path.exists():
                featured_df = pd.read_parquet(path)
                break

    if featured_df is None or featured_df.empty:
        logger.warning("No data available for graph analysis")
        return {}

    analyzer = CRENetworkAnalyzer()

    # Build ownership graph
    G = analyzer.build_ownership_graph(featured_df)

    results = {
        "n_nodes": G.number_of_nodes(),
        "n_edges": G.number_of_edges(),
    }

    if G.number_of_nodes() > 0:
        # Key players
        key_players = analyzer.identify_key_players()
        results["key_players"] = key_players.to_dict() if not key_players.empty else {}

        # Communities
        if G.number_of_nodes() > 10:
            communities = analyzer.detect_communities()
            results["communities"] = communities

        # Propensity to sell
        sell_df = analyzer.find_propensity_to_sell_signals(featured_df)
        results["propensity_summary"] = (
            sell_df["sell_likelihood"].value_counts().to_dict()
            if "sell_likelihood" in sell_df.columns else {}
        )

    return results


# ─────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="NYC CRE Investment Analytics Pipeline")
    parser.add_argument(
        "--step",
        choices=["acquire", "elt", "features", "model", "simulate", "graph", "all"],
        default="all",
        help="Pipeline step to run (default: all)",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Use demo data instead of live API calls",
    )
    parser.add_argument(
        "--data-scale",
        choices=sorted(DATA_SCALE_PROFILES.keys()),
        default=DEFAULT_DATA_SCALE,
        help=f"Data volume profile (default: {DEFAULT_DATA_SCALE})",
    )
    parser.add_argument(
        "--demo-size",
        type=int,
        default=DEMO_SAMPLE_SIZE,
        help=f"Demo sample size for synthetic data (default: {DEMO_SAMPLE_SIZE})",
    )
    parser.add_argument("--limit-pluto", type=int, default=None, help="Override PLUTO fetch limit")
    parser.add_argument(
        "--limit-sales",
        type=int,
        default=None,
        help="Override rolling sales fetch limit",
    )
    parser.add_argument(
        "--limit-permits",
        type=int,
        default=None,
        help="Override DOB permits fetch limit",
    )
    parser.add_argument(
        "--limit-violations",
        type=int,
        default=None,
        help="Override HPD violations fetch limit",
    )
    args = parser.parse_args()

    start_time = time.time()
    fetch_limits = _resolve_fetch_limits(
        data_scale=args.data_scale,
        limit_pluto=args.limit_pluto,
        limit_sales=args.limit_sales,
        limit_permits=args.limit_permits,
        limit_violations=args.limit_violations,
    )
    logger.info("🏢 NYC CRE Investment Analytics Pipeline")
    logger.info(
        f"   Step: {args.step} | Demo mode: {args.demo} | "
        f"Data scale: {args.data_scale}"
    )
    logger.info(f"   Effective fetch limits: {fetch_limits}")
    logger.info("=" * 60)

    try:
        if args.step in ["acquire", "all"]:
            raw_data = step_acquire(
                use_demo=args.demo,
                fetch_limits=fetch_limits,
                demo_size=max(args.demo_size, 1000),
            )
        else:
            raw_data = None

        if args.step in ["elt", "all"]:
            elt_data = step_elt(raw_data)
        else:
            elt_data = None

        if args.step in ["features", "all"]:
            featured_df = step_features(elt_data)
        else:
            featured_df = None

        macro_df = None
        macro_path = DATA_PROCESSED / "macro_economic.parquet"
        if macro_path.exists():
            macro_df = pd.read_parquet(macro_path)
        elif (DATA_RAW / "macro_economic.parquet").exists():
            macro_df = pd.read_parquet(DATA_RAW / "macro_economic.parquet")

        if args.step in ["model", "all"]:
            violations_for_models = None
            if isinstance(elt_data, dict):
                violations_for_models = elt_data.get("staging", {}).get("violations")
            model_results = step_models(featured_df, macro_df, violations_for_models)
            logger.info(f"Model results: {list(model_results.keys())}")

        if args.step in ["simulate", "all"]:
            sim_results = step_simulation()

        if args.step in ["graph", "all"]:
            graph_results = step_graph(featured_df)

        elapsed = time.time() - start_time
        logger.info("=" * 60)
        logger.info(f"✅ Pipeline complete in {elapsed:.1f}s")
        logger.info(f"   Processed data saved to: {DATA_PROCESSED}")
        logger.info(f"   Model artifacts saved to: {MODEL_ARTIFACTS}")
        logger.info(f"   Launch dashboard: streamlit run app/dashboard.py")

    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(f"❌ Pipeline failed after {elapsed:.1f}s: {e}")
        raise


if __name__ == "__main__":
    main()
