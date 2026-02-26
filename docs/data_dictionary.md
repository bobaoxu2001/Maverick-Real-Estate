# NYC CRE Investment Intelligence — Data Dictionary

Author: **Allen Xu**

This document defines the core analytical datasets, key columns, and business interpretation used across the pipeline and dashboard.

---

## 1) Raw Sources

### 1.1 PLUTO (`raw/pluto.parquet`)

| Column | Type | Description | Business Use |
|---|---|---|---|
| `bbl` | string | Borough-Block-Lot unique property ID | Primary join key |
| `borocode` | int | Borough code (1-5) | Borough normalization |
| `assesstot` | float | Total assessed value | Valuation baseline |
| `lotarea` | float | Lot area (sqft) | Land utilization |
| `bldgarea` | float | Building area (sqft) | FAR and intensity |
| `comarea` | float | Commercial area (sqft) | Asset type composition |
| `numfloors` | float | Number of floors | Vertical density |
| `yearbuilt` | float | Build year | Building age |
| `latitude`, `longitude` | float | Geospatial coordinates | Distance and mapping |

### 1.2 DOF Rolling Sales (`raw/rolling_sales.parquet`)

| Column | Type | Description | Business Use |
|---|---|---|---|
| `bbl` | string | Property ID | Merge to PLUTO |
| `sale_price` | float | Transaction value | Target variable |
| `sale_date` | datetime | Transaction date | Time series |
| `gross_square_feet` | float | Building area in sale file | Price/SF |
| `land_square_feet` | float | Land area in sale file | Density proxies |
| `borough` | int | Borough code | Segmentation |
| `neighborhood` | string | Neighborhood name | Submarket analysis |
| `building_class_category` | string | Building class grouping | Feature engineering |
| `zip_code` | string | ZIP | Permit activity linkage |

### 1.3 DOB Permits (`raw/dob_permits.parquet`)

| Column | Type | Description | Business Use |
|---|---|---|---|
| `filing_date` | datetime | Permit filing date | Development momentum |
| `job_type` | string | NB/A1/DM classification | New build vs alteration |
| `estimated_job_cost` | float | Estimated project cost | Capex intensity |
| `zip_code` | string | ZIP | Neighborhood-level signal |

### 1.4 HPD Violations (`raw/hpd_violations.parquet`)

| Column | Type | Description | Business Use |
|---|---|---|---|
| `inspectiondate` | datetime | Inspection date | Distress timing signal |
| `class` | string | Severity class | Risk scoring |
| `boroid` / `boroughid` | string | Borough identifier | Borough distress profile |
| `bbl` (if present) | string | Property ID | Property-level merge |

### 1.5 FRED Macro (`raw/macro_economic.parquet`)

| Column | Type | Description | Business Use |
|---|---|---|---|
| `fed_funds_rate` | float | Policy rate | Financing cost |
| `treasury_10y` | float | Long-term benchmark yield | Discount rate proxy |
| `cpi_urban` | float | Inflation index | Real return context |
| `unemployment_nyc` | float | NYC labor market slack | Demand pressure |
| `commercial_mortgage_delinquency` | float | Credit stress indicator | Downside risk |
| `cre_price_index` | float | National CRE price proxy | Cycle context |

---

## 2) Processed Analytical Tables

### 2.1 `processed/property_valuations.parquet`

Canonical property-level mart for valuation and risk models. Includes cleaned sale fields plus merged PLUTO attributes.

### 2.2 `processed/featured_properties.parquet`

Model-ready dataset after feature engineering.

Key engineered fields:

| Feature | Type | Meaning |
|---|---|---|
| `building_age` | float | Current year minus `yearbuilt` |
| `price_per_sqft` | float | `sale_price / gross_square_feet` |
| `floor_area_ratio` | float | `bldgarea / lotarea` |
| `commercial_ratio` | float | `comarea / bldgarea` |
| `is_recently_renovated` | int (0/1) | Renovated within recent window |
| `dist_to_*_km` | float | Distance to major landmarks |
| `dist_to_cbd_km` | float | Midtown CBD distance proxy |
| `neighborhood_avg_price` | float | Neighborhood-level mean sale price |
| `neighborhood_sales_volume` | int | Neighborhood transactions |
| `neighborhood_permit_activity` | int | Permit count signal |
| `borough_violation_count` | int | Borough-level violation pressure |
| `price_vs_neighborhood` | float | Relative valuation ratio |
| `neighborhood_target_encoded` | float | OOF target encoding |

### 2.3 `processed/clustered_properties.parquet`

Adds `cluster` label for market segmentation outputs.

### 2.4 `processed/scored_properties.parquet`

Adds distress model outputs:

| Column | Type | Description |
|---|---|---|
| `distress_probability` | float [0,1] | Calibrated default/distress probability |
| `risk_tier` | category | Low / Moderate / Elevated / High |

---

## 3) Data Quality Rules (Operational)

- Drop zero / invalid sale prices (`sale_price >= 100000`).
- Restrict unrealistic `price_per_sqft` outliers.
- Remove invalid coordinates outside NYC bounds.
- Use median imputation for sparse numeric fields.
- Drop features with extreme missingness (>70%).
- Use out-of-fold target encoding to reduce leakage in training features.

---

## 4) Update Cadence

- Transactional / permit / violation feeds: typically refreshed per pipeline run.
- Macro series: monthly.
- Recommended production refresh:
  - **Daily** ingestion for city datasets.
  - **Monthly** macro refresh.
  - **Weekly** model retrain and diagnostics report regeneration.
