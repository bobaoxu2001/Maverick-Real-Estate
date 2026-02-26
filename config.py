"""
Configuration for NYC CRE Investment Analytics Pipeline.
Centralizes API endpoints, dataset identifiers, and model parameters.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


def _env_int(name: str, default: int) -> int:
    """Parse a positive integer from environment variables."""
    value = os.getenv(name)
    if value is None:
        return default
    try:
        parsed = int(value)
        return parsed if parsed > 0 else default
    except ValueError:
        return default

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"
REPORTS_MODEL_DIR = REPORTS_DIR / "model_performance"
MODEL_ARTIFACTS = PROJECT_ROOT / "models"

for d in [DATA_RAW, DATA_PROCESSED, REPORTS_DIR, REPORTS_MODEL_DIR, MODEL_ARTIFACTS]:
    d.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────
# NYC Open Data (Socrata API)
# Docs: https://dev.socrata.com/
# ──────────────────────────────────────────────
NYC_OPENDATA_DOMAIN = "data.cityofnewyork.us"
NYC_APP_TOKEN = os.getenv("NYC_APP_TOKEN", None)  # Optional; increases rate limit

# Dataset IDs (Socrata 4x4 identifiers)
DATASETS = {
    # PLUTO — Primary Land Use Tax Lot Output (property characteristics)
    "pluto": "64uk-42ks",
    # DOF Rolling Sales — actual property sale transactions
    "rolling_sales": "usep-8jbt",
    # DOB Permits — building permits (development activity signal)
    "dob_permits": "ipu4-2vj7",
    # HPD Violations — housing violations (distress signal)
    "hpd_violations": "wvxf-dwi5",
    # DOF Property Valuation & Assessment — assessed values
    "property_valuation": "yjxr-fw8i",
    # MapPLUTO (geospatial) — lot boundaries with land use
    "mappluto": "evjd-dqpz",
}

# Data scale presets (Socrata API supports pagination, so limits may exceed 50k)
DATA_SCALE_PROFILES = {
    "quick": {
        "pluto": 50000,
        "rolling_sales": 50000,
        "dob_permits": 30000,
        "hpd_violations": 30000,
    },
    "professional": {
        "pluto": 150000,
        "rolling_sales": 150000,
        "dob_permits": 100000,
        "hpd_violations": 100000,
    },
    "institutional": {
        "pluto": 250000,
        "rolling_sales": 300000,
        "dob_permits": 200000,
        "hpd_violations": 200000,
    },
}
DEFAULT_DATA_SCALE = os.getenv("DATA_SCALE", "professional").lower()
if DEFAULT_DATA_SCALE not in DATA_SCALE_PROFILES:
    DEFAULT_DATA_SCALE = "professional"

DATA_FETCH_LIMITS = {
    "pluto": _env_int("LIMIT_PLUTO", DATA_SCALE_PROFILES[DEFAULT_DATA_SCALE]["pluto"]),
    "rolling_sales": _env_int(
        "LIMIT_ROLLING_SALES",
        DATA_SCALE_PROFILES[DEFAULT_DATA_SCALE]["rolling_sales"],
    ),
    "dob_permits": _env_int(
        "LIMIT_DOB_PERMITS",
        DATA_SCALE_PROFILES[DEFAULT_DATA_SCALE]["dob_permits"],
    ),
    "hpd_violations": _env_int(
        "LIMIT_HPD_VIOLATIONS",
        DATA_SCALE_PROFILES[DEFAULT_DATA_SCALE]["hpd_violations"],
    ),
}
DEMO_SAMPLE_SIZE = _env_int("DEMO_SAMPLE_SIZE", 12000)
MACRO_START_DATE = os.getenv("MACRO_START_DATE", "2005-01-01")

# ──────────────────────────────────────────────
# FRED API  (Federal Reserve Economic Data)
# ──────────────────────────────────────────────
FRED_API_KEY = os.getenv("FRED_API_KEY", None)

FRED_SERIES = {
    "fed_funds_rate": "FEDFUNDS",
    "treasury_10y": "DGS10",
    "cpi_urban": "CPIAUCSL",
    "unemployment_nyc": "NEWY636URN",
    "commercial_mortgage_delinquency": "DRCLACBS",
    "cre_price_index": "COMREPUSQ159N",
}

# ──────────────────────────────────────────────
# NYC Geography Constants
# ──────────────────────────────────────────────
NYC_CENTER = (40.7128, -74.0060)
BOROUGH_CODES = {
    1: "Manhattan",
    2: "Bronx",
    3: "Brooklyn",
    4: "Queens",
    5: "Staten Island",
}

# Key landmarks for geospatial feature engineering
LANDMARKS = {
    "grand_central": (40.7527, -73.9772),
    "penn_station": (40.7506, -73.9935),
    "world_trade_center": (40.7127, -74.0134),
    "central_park": (40.7829, -73.9654),
    "times_square": (40.7580, -73.9855),
    "hudson_yards": (40.7539, -74.0020),
}

# ──────────────────────────────────────────────
# Property Filters (focus on commercial)
# ──────────────────────────────────────────────
COMMERCIAL_BUILDING_CLASSES = [
    "O",  # Office
    "K",  # Store buildings
    "L",  # Loft buildings
    "F",  # Factory / industrial
    "E",  # Warehouses
    "H",  # Hotels
    "I",  # Hospitals & health facilities
    "J",  # Theatres
    "RR", # Condominiums (commercial)
]

# ──────────────────────────────────────────────
# Model Parameters (defaults)
# ──────────────────────────────────────────────
MODEL_PARAMS = {
    "hedonic_regression": {
        "test_size": 0.2,
        "random_state": 42,
        "cv_folds": 5,
        "alpha_range": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
    },
    "clustering": {
        "max_clusters": 12,
        "random_state": 42,
        "min_samples_dbscan": 10,
    },
    "time_series": {
        "forecast_periods": 24,  # months
        "seasonality_period": 12,
        "changepoint_prior_scale": 0.05,
    },
    "distress_predictor": {
        "test_size": 0.2,
        "random_state": 42,
        "n_estimators": 500,
        "max_depth": 6,
        "learning_rate": 0.05,
    },
    "simulation": {
        "n_simulations": 10000,
        "time_horizon_years": 5,
        "confidence_intervals": [0.05, 0.25, 0.50, 0.75, 0.95],
    },
}

# ──────────────────────────────────────────────
# Neo4j (Graph Database)
# ──────────────────────────────────────────────
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")

# ──────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
