"""
ELT Pipeline Orchestration
============================
Demonstrates a production-ready ELT pipeline pattern using Prefect for
orchestration, mirroring the dbt + BigQuery architecture described in
Maverick's data stack.

In production, this pipeline would:
  1. Extract data from APIs → staging tables in BigQuery
  2. Load via dbt models (staging → intermediate → marts)
  3. Transform using SQL in dbt (see dbt_models/ directory)

For this project, we implement a local equivalent using pandas with the
same logical flow.

Author: Allen Xu
"""

import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger
from datetime import datetime

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
from config import DATA_RAW, DATA_PROCESSED, BOROUGH_CODES


# ─────────────────────────────────────────────────────────
# Pipeline Steps (mirroring dbt model layers)
# ─────────────────────────────────────────────────────────

def staging_pluto(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Staging layer: Clean and standardize PLUTO data.

    Mirrors: dbt_models/models/staging/stg_pluto.sql
    """
    df = raw_df.copy()

    # Standardize column names
    df.columns = df.columns.str.lower().str.strip()

    # Type coercion
    numeric_cols = [
        "assesstot", "assessland", "lotarea", "bldgarea", "comarea",
        "resarea", "officearea", "retailarea", "numfloors", "unitstotal",
        "unitsres", "yearbuilt", "latitude", "longitude",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Remove invalid coordinates
    if "latitude" in df.columns:
        df = df[
            (df["latitude"].between(40.4, 41.0)) &
            (df["longitude"].between(-74.3, -73.6))
        ]

    # Add borough name
    if "borocode" in df.columns:
        df["borocode"] = pd.to_numeric(df["borocode"], errors="coerce")
        df["borough_name"] = df["borocode"].map(BOROUGH_CODES)

    # Remove properties with zero assessed value
    if "assesstot" in df.columns:
        df = df[df["assesstot"] > 0]

    # Deduplicate by BBL
    if "bbl" in df.columns:
        df = df.drop_duplicates(subset="bbl", keep="first")

    logger.info(f"stg_pluto: {len(df)} rows after cleaning")
    return df


def staging_sales(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Staging layer: Clean and standardize DOF Rolling Sales.

    Mirrors: dbt_models/models/staging/stg_sales.sql
    """
    df = raw_df.copy()
    df.columns = df.columns.str.lower().str.strip()

    # Type coercion
    for col in ["sale_price", "gross_square_feet", "land_square_feet"]:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col].astype(str).str.replace(",", "").str.replace("$", ""),
                errors="coerce",
            )

    # Parse sale date
    if "sale_date" in df.columns:
        df["sale_date"] = pd.to_datetime(df["sale_date"], errors="coerce")
        df["sale_year"] = df["sale_date"].dt.year
        df["sale_month"] = df["sale_date"].dt.month
        df["sale_quarter"] = df["sale_date"].dt.quarter

    # Filter out non-arms-length transactions
    # (very low prices indicate transfers, not market sales)
    if "sale_price" in df.columns:
        df = df[df["sale_price"] >= 100000]

    # Filter out unreasonable prices per sqft
    if "sale_price" in df.columns and "gross_square_feet" in df.columns:
        df["price_per_sqft"] = df["sale_price"] / df["gross_square_feet"].replace(0, np.nan)
        df = df[
            (df["price_per_sqft"].isna()) |
            ((df["price_per_sqft"] >= 50) & (df["price_per_sqft"] <= 50000))
        ]

    # Borough name
    if "borough" in df.columns:
        df["borough"] = pd.to_numeric(df["borough"], errors="coerce")
        df["borough_name"] = df["borough"].map(BOROUGH_CODES)

    logger.info(f"stg_sales: {len(df)} rows after cleaning")
    return df


def staging_permits(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Staging layer: Clean DOB Permits data.

    Mirrors: dbt_models/models/staging/stg_permits.sql
    """
    df = raw_df.copy()
    df.columns = df.columns.str.lower().str.strip()

    if "filing_date" in df.columns:
        df["filing_date"] = pd.to_datetime(df["filing_date"], errors="coerce")
        df["filing_year"] = df["filing_date"].dt.year

    if "estimated_job_cost" in df.columns:
        df["estimated_job_cost"] = pd.to_numeric(
            df["estimated_job_cost"].astype(str).str.replace(",", ""),
            errors="coerce",
        )

    logger.info(f"stg_permits: {len(df)} rows after cleaning")
    return df


def staging_violations(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Staging layer: Clean HPD Violations data.

    Mirrors: dbt_models/models/staging/stg_violations.sql
    """
    df = raw_df.copy()
    df.columns = df.columns.str.lower().str.strip()

    if "inspectiondate" in df.columns:
        df["inspectiondate"] = pd.to_datetime(df["inspectiondate"], errors="coerce")

    logger.info(f"stg_violations: {len(df)} rows after cleaning")
    return df


# ── Intermediate Layer ────────────────────────────────

def intermediate_property_features(
    stg_pluto: pd.DataFrame,
    stg_sales: pd.DataFrame,
) -> pd.DataFrame:
    """Intermediate layer: Merge PLUTO with sales to create property feature set.

    Mirrors: dbt_models/models/intermediate/int_property_features.sql
    """
    # Merge on BBL if available
    if "bbl" in stg_pluto.columns and "bbl" in stg_sales.columns:
        stg_pluto["bbl"] = stg_pluto["bbl"].astype(str).str.strip()
        stg_sales["bbl"] = stg_sales["bbl"].astype(str).str.strip()
        merged = stg_sales.merge(
            stg_pluto,
            on="bbl",
            how="left",
            suffixes=("", "_pluto"),
        )
    else:
        merged = stg_sales.copy()
        logger.warning("No BBL column — skipping PLUTO merge in intermediate layer")

    logger.info(f"int_property_features: {len(merged)} rows")
    return merged


# ── Marts Layer ───────────────────────────────────────

def mart_property_valuations(int_features: pd.DataFrame) -> pd.DataFrame:
    """Mart layer: Final table for property valuation models.

    Mirrors: dbt_models/models/marts/mart_property_valuations.sql
    """
    df = int_features.copy()

    # Select and rename final columns for modeling
    keep_cols = [
        "bbl", "borough_name", "neighborhood", "building_class_category",
        "sale_price", "sale_date", "sale_year",
        "gross_square_feet", "land_square_feet",
        "yearbuilt", "numfloors", "unitstotal",
        "assesstot", "assessland", "lotarea", "bldgarea",
        "comarea", "resarea", "officearea", "retailarea",
        "latitude", "longitude", "price_per_sqft",
        "zip_code",
    ]

    available = [c for c in keep_cols if c in df.columns]
    df = df[available].copy()

    logger.info(f"mart_property_valuations: {len(df)} rows, {len(df.columns)} columns")
    return df


def mart_market_segments(int_features: pd.DataFrame) -> pd.DataFrame:
    """Mart layer: Aggregated neighborhood-level data for market segmentation.

    Mirrors: dbt_models/models/marts/mart_market_segments.sql
    """
    df = int_features.copy()

    group_col = "neighborhood" if "neighborhood" in df.columns else "borough_name"
    if group_col not in df.columns:
        logger.warning(f"No grouping column found for market segments")
        return pd.DataFrame()

    agg_dict = {}
    if "sale_price" in df.columns:
        agg_dict["sale_price"] = ["mean", "median", "count", "std"]
    if "price_per_sqft" in df.columns:
        agg_dict["price_per_sqft"] = ["mean", "median"]
    if "yearbuilt" in df.columns:
        agg_dict["yearbuilt"] = "median"
    if "numfloors" in df.columns:
        agg_dict["numfloors"] = "mean"

    if not agg_dict:
        return pd.DataFrame()

    market = df.groupby(group_col).agg(agg_dict)
    market.columns = ["_".join(col).strip("_") for col in market.columns]
    market = market.reset_index()

    logger.info(f"mart_market_segments: {len(market)} segments")
    return market


# ─────────────────────────────────────────────────────────
# Full ELT Pipeline
# ─────────────────────────────────────────────────────────
def run_elt_pipeline(raw_data: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Execute the full ELT pipeline.

    Layers:
      1. Staging — clean raw data
      2. Intermediate — merge and enrich
      3. Marts — final analytical tables

    Args:
        raw_data: Dict of raw DataFrames from data acquisition

    Returns:
        Dict of processed DataFrames ready for modeling
    """
    logger.info("=" * 60)
    logger.info("Starting ELT Pipeline")
    logger.info("=" * 60)

    # ── Staging ──
    stg = {}
    if "pluto" in raw_data:
        stg["pluto"] = staging_pluto(raw_data["pluto"])
    if "rolling_sales" in raw_data:
        stg["sales"] = staging_sales(raw_data["rolling_sales"])
    if "dob_permits" in raw_data:
        stg["permits"] = staging_permits(raw_data["dob_permits"])
    if "hpd_violations" in raw_data:
        stg["violations"] = staging_violations(raw_data["hpd_violations"])

    # ── Intermediate ──
    int_features = None
    if "pluto" in stg and "sales" in stg:
        int_features = intermediate_property_features(stg["pluto"], stg["sales"])
    elif "sales" in stg:
        int_features = stg["sales"]

    # ── Marts ──
    results = {"staging": stg}

    if int_features is not None:
        results["property_valuations"] = mart_property_valuations(int_features)
        results["market_segments"] = mart_market_segments(int_features)
    else:
        logger.warning("No data available for mart layer")

    # Include macro data if available
    if "macro_economic" in raw_data:
        results["macro_economic"] = raw_data["macro_economic"]

    # Save processed data
    for name, df in results.items():
        if isinstance(df, pd.DataFrame):
            path = DATA_PROCESSED / f"{name}.parquet"
            df.to_parquet(path, index=True)
            logger.info(f"Saved {name} → {path}")

    logger.info("ELT Pipeline complete!")
    return results
