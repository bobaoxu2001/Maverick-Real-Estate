"""
Feature Engineering Module
===========================
Transforms raw NYC CRE data into model-ready features.

Covers:
  - Geospatial features (distances to landmarks, subway proximity)
  - Property characteristic encoding (building class, zoning, age)
  - Market context features (neighborhood trends, permit activity)
  - Dimensionality reduction (PCA)
  - Handling of sparseness and categorical encoding

Author: Allen Xu
"""

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.preprocessing import (
    StandardScaler,
    LabelEncoder,
    OneHotEncoder,
)
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from loguru import logger

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
from config import LANDMARKS, NYC_CENTER, BOROUGH_CODES


# ─────────────────────────────────────────────────────────
# Geospatial Features
# ─────────────────────────────────────────────────────────
def haversine_distance(lat1, lon1, lat2, lon2):
    """Vectorized haversine distance in kilometers."""
    R = 6371  # Earth radius in km
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def add_geospatial_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add distance-based features from property coordinates to key NYC landmarks.

    Features created:
      - dist_to_{landmark}_km: Distance to each landmark
      - dist_to_cbd_km: Distance to Midtown CBD (Grand Central)
      - min_landmark_dist_km: Minimum distance to any landmark
    """
    df = df.copy()

    if "latitude" not in df.columns or "longitude" not in df.columns:
        logger.warning("No lat/lon columns found — skipping geospatial features")
        return df

    lat = df["latitude"].astype(float)
    lon = df["longitude"].astype(float)

    for name, (lm_lat, lm_lon) in LANDMARKS.items():
        col_name = f"dist_to_{name}_km"
        df[col_name] = haversine_distance(lat, lon, lm_lat, lm_lon)

    # CBD distance (Grand Central as proxy for Midtown CBD)
    df["dist_to_cbd_km"] = df["dist_to_grand_central_km"]

    # Minimum distance to any key landmark
    dist_cols = [c for c in df.columns if c.startswith("dist_to_") and c != "dist_to_cbd_km"]
    df["min_landmark_dist_km"] = df[dist_cols].min(axis=1)

    # Manhattan premium: binary indicator
    if "borocode" in df.columns:
        df["is_manhattan"] = (pd.to_numeric(df["borocode"], errors="coerce") == 1).astype(int)

    logger.info(f"Added {len(LANDMARKS) + 2} geospatial features")
    return df


# ─────────────────────────────────────────────────────────
# Property Characteristic Features
# ─────────────────────────────────────────────────────────
def add_property_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer property-level features from PLUTO / sales data.

    Features:
      - building_age: Current year minus year built
      - price_per_sqft: Sale price / gross square feet
      - lot_building_ratio: Building area / lot area (FAR proxy)
      - commercial_ratio: Commercial area / total building area
      - is_recently_renovated: Whether alterations occurred in last 15 years
      - log_assesstot: Log-transformed total assessed value
      - floor_area_ratio: Total building area / lot area
    """
    df = df.copy()
    current_year = pd.Timestamp.now().year

    # Building age
    if "yearbuilt" in df.columns:
        df["yearbuilt"] = pd.to_numeric(df["yearbuilt"], errors="coerce")
        df["building_age"] = current_year - df["yearbuilt"]
        df.loc[df["building_age"] < 0, "building_age"] = np.nan
        df.loc[df["building_age"] > 300, "building_age"] = np.nan

    # Price per square foot
    if "sale_price" in df.columns and "gross_square_feet" in df.columns:
        df["sale_price"] = pd.to_numeric(df["sale_price"], errors="coerce")
        df["gross_square_feet"] = pd.to_numeric(df["gross_square_feet"], errors="coerce")
        df["price_per_sqft"] = df["sale_price"] / df["gross_square_feet"].replace(0, np.nan)

    # Floor Area Ratio (FAR) — key CRE metric
    if "bldgarea" in df.columns and "lotarea" in df.columns:
        df["bldgarea"] = pd.to_numeric(df["bldgarea"], errors="coerce")
        df["lotarea"] = pd.to_numeric(df["lotarea"], errors="coerce")
        df["floor_area_ratio"] = df["bldgarea"] / df["lotarea"].replace(0, np.nan)

    # Commercial area ratio
    if "comarea" in df.columns and "bldgarea" in df.columns:
        df["comarea"] = pd.to_numeric(df["comarea"], errors="coerce")
        df["commercial_ratio"] = df["comarea"] / df["bldgarea"].replace(0, np.nan)

    # Recently renovated indicator
    if "yearalter1" in df.columns:
        df["yearalter1"] = pd.to_numeric(df["yearalter1"], errors="coerce")
        df["is_recently_renovated"] = (
            (current_year - df["yearalter1"]) <= 15
        ).astype(int)
        df.loc[df["yearalter1"].isna(), "is_recently_renovated"] = 0

    # Log transformations for skewed distributions
    for col in ["assesstot", "assessland", "bldgarea", "lotarea"]:
        if col in df.columns:
            vals = pd.to_numeric(df[col], errors="coerce")
            df[f"log_{col}"] = np.log1p(vals.clip(lower=0))

    # Number of floors bucketed
    if "numfloors" in df.columns:
        df["numfloors"] = pd.to_numeric(df["numfloors"], errors="coerce")
        df["height_category"] = pd.cut(
            df["numfloors"],
            bins=[0, 5, 15, 30, 60, 200],
            labels=["low_rise", "mid_rise", "high_rise", "tower", "supertall"],
        )

    logger.info("Added property characteristic features")
    return df


# ─────────────────────────────────────────────────────────
# Market Context Features
# ─────────────────────────────────────────────────────────
def add_market_context_features(
    sales_df: pd.DataFrame,
    permits_df: pd.DataFrame,
    violations_df: pd.DataFrame,
) -> pd.DataFrame:
    """Add neighborhood-level market context to sales data.

    Features:
      - neighborhood_avg_price: Average sale price in neighborhood
      - neighborhood_sales_volume: Number of recent sales
      - neighborhood_permit_activity: Count of recent building permits
      - neighborhood_violation_rate: HPD violations per property
      - price_vs_neighborhood: Ratio of property price to neighborhood average
    """
    df = sales_df.copy()

    # Neighborhood average price and volume
    if "neighborhood" in df.columns and "sale_price" in df.columns:
        neighborhood_stats = df.groupby("neighborhood").agg(
            neighborhood_avg_price=("sale_price", "mean"),
            neighborhood_median_price=("sale_price", "median"),
            neighborhood_sales_volume=("sale_price", "count"),
            neighborhood_price_std=("sale_price", "std"),
        ).reset_index()

        df = df.merge(neighborhood_stats, on="neighborhood", how="left")
        df["price_vs_neighborhood"] = df["sale_price"] / df["neighborhood_avg_price"].replace(0, np.nan)

    # Permit activity by zip code (development signal)
    if permits_df is not None and "zip_code" in df.columns:
        if "zip_code" in permits_df.columns:
            permit_counts = (
                permits_df.groupby("zip_code")
                .size()
                .reset_index(name="neighborhood_permit_activity")
            )
            # Normalize zip codes for merge
            df["zip_code"] = df["zip_code"].astype(str).str.strip()
            permit_counts["zip_code"] = permit_counts["zip_code"].astype(str).str.strip()
            df = df.merge(permit_counts, on="zip_code", how="left")
            df["neighborhood_permit_activity"] = df["neighborhood_permit_activity"].fillna(0)

    # Violation rate (distress signal)
    if violations_df is not None and "boroid" in violations_df.columns:
        if "borough" in df.columns:
            violation_counts = (
                violations_df.groupby("boroid")
                .size()
                .reset_index(name="borough_violation_count")
            )
            violation_counts["boroid"] = violation_counts["boroid"].astype(str)
            df["borough"] = df["borough"].astype(str)
            df = df.merge(
                violation_counts,
                left_on="borough",
                right_on="boroid",
                how="left",
            )
            df["borough_violation_count"] = df["borough_violation_count"].fillna(0)

    logger.info("Added market context features")
    return df


# ─────────────────────────────────────────────────────────
# Categorical Encoding
# ─────────────────────────────────────────────────────────
def encode_categoricals(
    df: pd.DataFrame,
    one_hot_cols: list[str] = None,
    label_cols: list[str] = None,
    target_encode_cols: list[str] = None,
    target_col: str = "sale_price",
) -> pd.DataFrame:
    """Encode categorical variables using multiple strategies.

    Strategies:
      - One-hot: For low-cardinality categories (e.g., borough)
      - Label encoding: For ordinal categories (e.g., building class)
      - Target encoding: For high-cardinality categories (e.g., neighborhood)
    """
    df = df.copy()

    # One-hot encoding
    if one_hot_cols:
        for col in one_hot_cols:
            if col in df.columns:
                dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
                df = pd.concat([df, dummies], axis=1)
                df.drop(col, axis=1, inplace=True)
                logger.debug(f"One-hot encoded: {col} → {len(dummies.columns)} columns")

    # Label encoding
    if label_cols:
        encoders = {}
        for col in label_cols:
            if col in df.columns:
                le = LabelEncoder()
                mask = df[col].notna()
                df.loc[mask, f"{col}_encoded"] = le.fit_transform(
                    df.loc[mask, col].astype(str)
                )
                encoders[col] = le
                logger.debug(f"Label encoded: {col}")

    # Target encoding (mean encoding with smoothing)
    if target_encode_cols and target_col in df.columns:
        global_mean = df[target_col].mean()
        smoothing = 10

        for col in target_encode_cols:
            if col in df.columns:
                agg = df.groupby(col)[target_col].agg(["mean", "count"])
                smooth = (agg["count"] * agg["mean"] + smoothing * global_mean) / (
                    agg["count"] + smoothing
                )
                df[f"{col}_target_encoded"] = df[col].map(smooth)
                logger.debug(f"Target encoded: {col}")

    return df


# ─────────────────────────────────────────────────────────
# Dimensionality Reduction
# ─────────────────────────────────────────────────────────
def apply_pca(
    df: pd.DataFrame,
    feature_cols: list[str],
    n_components: int = 10,
    variance_threshold: float = 0.95,
) -> tuple[pd.DataFrame, PCA, StandardScaler]:
    """Apply PCA for dimensionality reduction on numeric features.

    Returns:
      - DataFrame with PCA components added
      - Fitted PCA object (for interpretation)
      - Fitted scaler (for transforms on new data)
    """
    # Select and impute numeric features
    available_cols = [c for c in feature_cols if c in df.columns]
    X = df[available_cols].copy()

    imputer = SimpleImputer(strategy="median")
    X_imputed = pd.DataFrame(
        imputer.fit_transform(X), columns=available_cols, index=X.index
    )

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    # Determine n_components from variance threshold
    pca_full = PCA().fit(X_scaled)
    cumvar = np.cumsum(pca_full.explained_variance_ratio_)
    n_auto = int(np.argmax(cumvar >= variance_threshold) + 1)
    n_components = min(n_components, n_auto, X_scaled.shape[1])

    # Fit PCA
    pca = PCA(n_components=n_components)
    components = pca.fit_transform(X_scaled)

    # Add components to DataFrame
    pca_cols = [f"pca_{i+1}" for i in range(n_components)]
    pca_df = pd.DataFrame(components, columns=pca_cols, index=df.index)
    df_out = pd.concat([df, pca_df], axis=1)

    logger.info(
        f"PCA: {len(available_cols)} features → {n_components} components "
        f"(explained variance: {pca.explained_variance_ratio_.sum():.1%})"
    )

    return df_out, pca, scaler


# ─────────────────────────────────────────────────────────
# Sparseness Handling
# ─────────────────────────────────────────────────────────
def handle_sparseness(df: pd.DataFrame, threshold: float = 0.7) -> pd.DataFrame:
    """Handle sparse features by dropping or imputing.

    - Drops columns where more than `threshold` fraction of values are missing.
    - Imputes remaining numeric columns with median.
    - Imputes remaining categorical columns with mode.
    """
    df = df.copy()
    n_rows = len(df)

    # Drop highly sparse columns
    missing_frac = df.isnull().mean()
    drop_cols = missing_frac[missing_frac > threshold].index.tolist()
    if drop_cols:
        logger.info(f"Dropping {len(drop_cols)} columns with >{threshold:.0%} missing: {drop_cols[:5]}...")
        df = df.drop(columns=drop_cols)

    # Impute remaining
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    categorical_cols = df.select_dtypes(include=["object", "category"]).columns
    for col in categorical_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].mode().iloc[0] if len(df[col].mode()) > 0 else "Unknown")

    logger.info(f"Sparseness handling: {len(drop_cols)} cols dropped, remaining imputed")
    return df


# ─────────────────────────────────────────────────────────
# Full Feature Engineering Pipeline
# ─────────────────────────────────────────────────────────
def engineer_features(
    pluto_df: pd.DataFrame,
    sales_df: pd.DataFrame,
    permits_df: pd.DataFrame = None,
    violations_df: pd.DataFrame = None,
) -> pd.DataFrame:
    """Run the full feature engineering pipeline.

    Steps:
      1. Merge PLUTO property characteristics with sales data
      2. Add geospatial features
      3. Add property-level features
      4. Add market context features
      5. Encode categorical variables
      6. Handle sparseness
    """
    logger.info("Starting feature engineering pipeline...")

    # ── Step 1: Merge PLUTO with Sales ──
    # Try to merge on BBL (borough-block-lot) which is the standard NYC property ID
    if "bbl" in pluto_df.columns and "bbl" in sales_df.columns:
        merged = sales_df.merge(
            pluto_df.drop_duplicates(subset="bbl"),
            on="bbl",
            how="left",
            suffixes=("", "_pluto"),
        )
    else:
        # Fallback: use sales data only
        merged = sales_df.copy()
        logger.warning("No BBL column found — using sales data only without PLUTO merge")

    # ── Step 2: Geospatial Features ──
    merged = add_geospatial_features(merged)

    # ── Step 3: Property Features ──
    merged = add_property_features(merged)

    # ── Step 4: Market Context ──
    merged = add_market_context_features(merged, permits_df, violations_df)

    # ── Step 5: Categorical Encoding ──
    merged = encode_categoricals(
        merged,
        one_hot_cols=["borough_name"],
        label_cols=["building_class_category"],
        target_encode_cols=["neighborhood"],
        target_col="sale_price",
    )

    # ── Step 6: Sparseness ──
    merged = handle_sparseness(merged, threshold=0.7)

    logger.info(f"Feature engineering complete: {merged.shape}")
    return merged
