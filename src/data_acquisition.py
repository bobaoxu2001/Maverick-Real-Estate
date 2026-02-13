"""
Data Acquisition Module
=======================
Pulls real-time and historical NYC commercial real estate data from public APIs:
  - NYC Open Data (Socrata) — PLUTO, DOF Sales, DOB Permits, HPD Violations
  - FRED — Macroeconomic indicators (interest rates, CPI, CRE indices)

Author: Allen Xu
"""

import time
from typing import Optional

import pandas as pd
import numpy as np
from loguru import logger
from sodapy import Socrata
from fredapi import Fred

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
from config import (
    NYC_OPENDATA_DOMAIN,
    NYC_APP_TOKEN,
    DATASETS,
    FRED_API_KEY,
    FRED_SERIES,
    BOROUGH_CODES,
    DATA_RAW,
)


# ─────────────────────────────────────────────────────────
# NYC Open Data Client
# ─────────────────────────────────────────────────────────
class NYCOpenDataClient:
    """Client for the NYC Open Data Socrata API.

    Provides typed accessors for each CRE-relevant dataset with built-in
    pagination, retry logic, and data type coercion.
    """

    def __init__(self, app_token: Optional[str] = None, timeout: int = 30):
        self.client = Socrata(
            NYC_OPENDATA_DOMAIN,
            app_token or NYC_APP_TOKEN,
            timeout=timeout,
        )
        logger.info(f"Initialized NYC Open Data client (domain={NYC_OPENDATA_DOMAIN})")

    def _fetch_dataset(
        self,
        dataset_id: str,
        limit: int = 50000,
        where: Optional[str] = None,
        select: Optional[str] = None,
        order: Optional[str] = None,
        offset: int = 0,
    ) -> pd.DataFrame:
        """Generic paginated fetch from Socrata with retry."""
        all_records = []
        current_offset = offset
        page_size = min(limit, 50000)
        retries = 3

        while True:
            for attempt in range(retries):
                try:
                    params = {"limit": page_size, "offset": current_offset}
                    if where:
                        params["where"] = where
                    if select:
                        params["select"] = select
                    if order:
                        params["order"] = order

                    records = self.client.get(dataset_id, **params)
                    break
                except Exception as e:
                    if attempt < retries - 1:
                        wait = 2 ** (attempt + 1)
                        logger.warning(f"Retry {attempt+1}/{retries} after error: {e}. Waiting {wait}s")
                        time.sleep(wait)
                    else:
                        raise

            if not records:
                break

            all_records.extend(records)
            logger.debug(f"Fetched {len(records)} records (total: {len(all_records)})")

            if len(records) < page_size or len(all_records) >= limit:
                break

            current_offset += page_size

        df = pd.DataFrame.from_records(all_records)
        logger.info(f"Dataset {dataset_id}: {len(df)} rows, {len(df.columns)} columns")
        return df

    # ── PLUTO (Property Land Use) ──────────────────────

    def fetch_pluto(self, borough: Optional[int] = None, limit: int = 50000) -> pd.DataFrame:
        """Fetch PLUTO data — property characteristics for every tax lot in NYC.

        Fields include: building class, zoning, lot area, building area,
        number of floors, year built, assessed value, coordinates, etc.
        """
        where_clauses = []
        # Focus on commercial properties
        where_clauses.append(
            "landuse IN('05', '04', '03', '02', '06', '07', '08', '09', '10', '11')"
        )
        if borough:
            where_clauses.append(f"borocode='{borough}'")

        where = " AND ".join(where_clauses) if where_clauses else None

        df = self._fetch_dataset(
            DATASETS["pluto"],
            limit=limit,
            where=where,
            order="assesstot DESC",
        )

        # Type coercion for numeric fields
        numeric_cols = [
            "assesstot", "assessland", "lotarea", "bldgarea", "comarea",
            "resarea", "officearea", "retailarea", "numfloors", "unitstotal",
            "unitsres", "yearbuilt", "yearalter1", "yearalter2",
            "latitude", "longitude",
        ]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        logger.info(f"PLUTO: {len(df)} commercial properties fetched")
        return df

    # ── DOF Rolling Sales ──────────────────────────────

    def fetch_rolling_sales(self, limit: int = 50000) -> pd.DataFrame:
        """Fetch DOF Rolling Sales — actual property sale transactions.

        Contains sale price, sale date, building class, neighborhood,
        tax class, gross/land square footage, etc.
        """
        # Filter for commercial sales with valid prices
        where = "sale_price > '100000'"

        df = self._fetch_dataset(
            DATASETS["rolling_sales"],
            limit=limit,
            where=where,
            order="sale_date DESC",
        )

        # Type coercion
        numeric_cols = [
            "sale_price", "gross_square_feet", "land_square_feet",
            "residential_units", "commercial_units", "total_units",
            "year_built",
        ]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        if "sale_date" in df.columns:
            df["sale_date"] = pd.to_datetime(df["sale_date"], errors="coerce")

        # Map borough codes
        if "borough" in df.columns:
            df["borough"] = pd.to_numeric(df["borough"], errors="coerce")
            df["borough_name"] = df["borough"].map(BOROUGH_CODES)

        logger.info(f"Rolling Sales: {len(df)} transactions fetched")
        return df

    # ── DOB Permits ────────────────────────────────────

    def fetch_dob_permits(self, limit: int = 30000) -> pd.DataFrame:
        """Fetch DOB building permits — development activity indicators.

        High permit activity signals neighborhood growth and can
        influence property values and investment attractiveness.
        """
        # Focus on recent permits with estimated cost
        where = (
            "filing_date > '2020-01-01T00:00:00.000' "
            "AND job_type IN('NB', 'A1', 'DM')"
        )

        df = self._fetch_dataset(
            DATASETS["dob_permits"],
            limit=limit,
            where=where,
            order="filing_date DESC",
        )

        if "filing_date" in df.columns:
            df["filing_date"] = pd.to_datetime(df["filing_date"], errors="coerce")
        if "issuance_date" in df.columns:
            df["issuance_date"] = pd.to_datetime(df["issuance_date"], errors="coerce")

        numeric_cols = ["estimated_job_cost", "initial_cost"]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(
                    df[col].astype(str).str.replace(",", ""), errors="coerce"
                )

        logger.info(f"DOB Permits: {len(df)} permits fetched")
        return df

    # ── HPD Violations ─────────────────────────────────

    def fetch_hpd_violations(self, limit: int = 30000) -> pd.DataFrame:
        """Fetch HPD violations — property distress signals.

        High violation counts indicate deferred maintenance and can
        signal financial distress or default risk.
        """
        where = "inspectiondate > '2020-01-01T00:00:00.000'"

        df = self._fetch_dataset(
            DATASETS["hpd_violations"],
            limit=limit,
            where=where,
            order="inspectiondate DESC",
        )

        if "inspectiondate" in df.columns:
            df["inspectiondate"] = pd.to_datetime(df["inspectiondate"], errors="coerce")

        logger.info(f"HPD Violations: {len(df)} violations fetched")
        return df

    # ── Convenience: Fetch All ─────────────────────────

    def fetch_all(self, limits: Optional[dict] = None) -> dict[str, pd.DataFrame]:
        """Fetch all CRE-relevant datasets and return as a dict."""
        defaults = {
            "pluto": 50000,
            "rolling_sales": 50000,
            "dob_permits": 30000,
            "hpd_violations": 30000,
        }
        limits = {**defaults, **(limits or {})}

        data = {}
        data["pluto"] = self.fetch_pluto(limit=limits["pluto"])
        data["rolling_sales"] = self.fetch_rolling_sales(limit=limits["rolling_sales"])
        data["dob_permits"] = self.fetch_dob_permits(limit=limits["dob_permits"])
        data["hpd_violations"] = self.fetch_hpd_violations(limit=limits["hpd_violations"])

        return data


# ─────────────────────────────────────────────────────────
# FRED Economic Data Client
# ─────────────────────────────────────────────────────────
class FREDClient:
    """Client for the Federal Reserve Economic Data API.

    Fetches macroeconomic indicators relevant to CRE investment:
    interest rates, inflation, unemployment, CRE-specific indices.
    """

    def __init__(self, api_key: Optional[str] = None):
        key = api_key or FRED_API_KEY
        if not key:
            logger.warning(
                "FRED_API_KEY not set. Macroeconomic data will use synthetic fallback. "
                "Get a free key at https://fred.stlouisfed.org/docs/api/api_key.html"
            )
            self.fred = None
        else:
            self.fred = Fred(api_key=key)
            logger.info("Initialized FRED API client")

    def fetch_series(
        self,
        series_id: str,
        start_date: str = "2010-01-01",
        end_date: Optional[str] = None,
    ) -> pd.Series:
        """Fetch a single FRED time series."""
        if self.fred is None:
            return self._synthetic_series(series_id, start_date, end_date)

        try:
            data = self.fred.get_series(
                series_id,
                observation_start=start_date,
                observation_end=end_date,
            )
            logger.info(f"FRED {series_id}: {len(data)} observations")
            return data
        except Exception as e:
            logger.error(f"Failed to fetch FRED {series_id}: {e}")
            return self._synthetic_series(series_id, start_date, end_date)

    def fetch_all_macro(
        self, start_date: str = "2010-01-01"
    ) -> pd.DataFrame:
        """Fetch all configured macroeconomic series into a single DataFrame."""
        series_dict = {}
        for name, series_id in FRED_SERIES.items():
            series_dict[name] = self.fetch_series(series_id, start_date)

        df = pd.DataFrame(series_dict)
        df.index.name = "date"

        # Forward-fill monthly/quarterly series to daily, then resample to monthly
        df = df.ffill()
        df_monthly = df.resample("MS").last()

        logger.info(f"Macro data: {len(df_monthly)} months, {len(df_monthly.columns)} series")
        return df_monthly

    @staticmethod
    def _synthetic_series(
        series_id: str, start_date: str, end_date: Optional[str]
    ) -> pd.Series:
        """Generate plausible synthetic data when API key is unavailable.

        Uses realistic ranges based on historical data (2010-2024).
        """
        rng = np.random.default_rng(hash(series_id) % (2**31))
        dates = pd.date_range(start=start_date, end=end_date or "2024-12-01", freq="MS")

        synthetic_params = {
            "FEDFUNDS": (1.5, 2.0),
            "DGS10": (2.5, 1.0),
            "CPIAUCSL": (260, 30),
            "NEWY636URN": (5.5, 2.0),
            "DRCLACBS": (1.5, 0.8),
            "COMREPUSQ159N": (400, 50),
        }

        mean, std = synthetic_params.get(series_id, (100, 10))
        trend = np.linspace(0, std * 0.5, len(dates))
        noise = rng.normal(0, std * 0.1, len(dates))
        values = mean + trend + np.cumsum(noise) * 0.3

        logger.warning(f"Using synthetic data for {series_id} (no API key)")
        return pd.Series(values, index=dates, name=series_id)


# ─────────────────────────────────────────────────────────
# Data Persistence
# ─────────────────────────────────────────────────────────
def save_raw_data(data: dict[str, pd.DataFrame], directory=DATA_RAW):
    """Save fetched DataFrames to parquet files."""
    for name, df in data.items():
        path = directory / f"{name}.parquet"
        df.to_parquet(path, index=False)
        logger.info(f"Saved {name} → {path} ({len(df)} rows)")


def load_raw_data(directory=DATA_RAW) -> dict[str, pd.DataFrame]:
    """Load previously saved raw data."""
    data = {}
    for path in directory.glob("*.parquet"):
        name = path.stem
        data[name] = pd.read_parquet(path)
        logger.info(f"Loaded {name} ← {path} ({len(data[name])} rows)")
    return data


# ─────────────────────────────────────────────────────────
# Main (standalone execution)
# ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    logger.info("Starting data acquisition pipeline...")

    # 1. NYC Open Data
    nyc_client = NYCOpenDataClient()
    nyc_data = nyc_client.fetch_all()

    # 2. FRED Macroeconomic Data
    fred_client = FREDClient()
    macro_df = fred_client.fetch_all_macro()
    nyc_data["macro_economic"] = macro_df

    # 3. Save
    save_raw_data(nyc_data)

    logger.success("Data acquisition complete!")
    for name, df in nyc_data.items():
        print(f"  {name}: {df.shape}")
