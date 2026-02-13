"""
Property Distress / Default Probability Predictor
===================================================
Predicts the likelihood that a commercial property will experience
financial distress, default, or forced sale — a critical signal for
both risk management and opportunistic investment.

Distress Signals:
  - High violation counts (deferred maintenance)
  - Declining neighborhood trends
  - High leverage relative to market value
  - Aging buildings without renovation
  - Rising vacancy (proxied by commercial ratio changes)
  - Macro headwinds (rising rates, tightening credit)

Model: XGBoost classifier with calibrated probabilities.

Author: Allen Xu
"""

import numpy as np
import pandas as pd
import joblib
from loguru import logger
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    roc_auc_score,
    precision_recall_curve,
    classification_report,
    average_precision_score,
    roc_curve,
    f1_score,
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent.parent))
from config import MODEL_PARAMS, MODEL_ARTIFACTS


class DistressPredictor:
    """Predicts commercial property distress/default probability.

    Uses property characteristics, market conditions, and violation
    history to estimate the probability that a property will experience
    financial distress within a given time horizon.

    The model is calibrated to produce reliable probability estimates,
    enabling risk-adjusted investment scoring.
    """

    DISTRESS_FEATURES = [
        # Property characteristics
        "building_age",
        "floor_area_ratio",
        "numfloors",
        "is_recently_renovated",
        "assesstot",
        "log_assesstot",

        # Location
        "dist_to_cbd_km",
        "is_manhattan",

        # Market context
        "price_vs_neighborhood",
        "neighborhood_sales_volume",
        "neighborhood_permit_activity",

        # Valuation signals
        "price_per_sqft",

        # Violation signals (if merged)
        "violation_count",
        "severe_violation_count",
    ]

    def __init__(self, params: dict = None):
        self.params = params or MODEL_PARAMS["distress_predictor"]
        self.pipeline = None
        self.calibrated_model = None
        self.feature_names = []
        self.results = {}

    def create_distress_labels(
        self,
        property_df: pd.DataFrame,
        violations_df: pd.DataFrame = None,
    ) -> pd.DataFrame:
        """Create binary distress labels from observable signals.

        Distress indicators (combined scoring):
          1. Property sold at >30% below neighborhood average
          2. High violation count (top quartile)
          3. Building age > 60 years with no renovation
          4. Extreme price decline (year-over-year)

        In production, this would be informed by actual default data
        from CMBS, bank records, or Maverick's proprietary datasets.
        """
        df = property_df.copy()

        # Initialize distress score
        df["distress_score"] = 0

        # Signal 1: Below-market sale price
        if "price_vs_neighborhood" in df.columns:
            df.loc[
                pd.to_numeric(df["price_vs_neighborhood"], errors="coerce") < 0.7,
                "distress_score",
            ] += 1

        # Signal 2: High violation count
        if violations_df is not None and "bbl" in violations_df.columns:
            violation_counts = violations_df.groupby("bbl").size().reset_index(name="violation_count")
            # Severe violations
            if "class" in violations_df.columns:
                severe = violations_df[violations_df["class"].isin(["C", "I"])]
                severe_counts = severe.groupby("bbl").size().reset_index(name="severe_violation_count")
                violation_counts = violation_counts.merge(severe_counts, on="bbl", how="left")
                violation_counts["severe_violation_count"] = violation_counts["severe_violation_count"].fillna(0)

            if "bbl" in df.columns:
                df = df.merge(violation_counts, on="bbl", how="left")
                df["violation_count"] = df.get("violation_count", pd.Series(0, index=df.index)).fillna(0)
                df["severe_violation_count"] = df.get("severe_violation_count", pd.Series(0, index=df.index)).fillna(0)

                # Top quartile violations = distress signal
                threshold = df["violation_count"].quantile(0.75)
                if threshold > 0:
                    df.loc[df["violation_count"] >= threshold, "distress_score"] += 1
        else:
            df["violation_count"] = 0
            df["severe_violation_count"] = 0

        # Signal 3: Aging building without renovation
        if "building_age" in df.columns and "is_recently_renovated" in df.columns:
            df.loc[
                (pd.to_numeric(df["building_age"], errors="coerce") > 60)
                & (df["is_recently_renovated"] == 0),
                "distress_score",
            ] += 1

        # Signal 4: Very low price per sqft (bottom 10%)
        if "price_per_sqft" in df.columns:
            ppsf = pd.to_numeric(df["price_per_sqft"], errors="coerce")
            threshold = ppsf.quantile(0.10)
            if threshold > 0:
                df.loc[ppsf <= threshold, "distress_score"] += 1

        # Binary label: distress = score >= 2
        df["is_distressed"] = (df["distress_score"] >= 2).astype(int)

        distress_rate = df["is_distressed"].mean()
        logger.info(
            f"Distress labels created: {df['is_distressed'].sum()} distressed "
            f"({distress_rate:.1%} of {len(df)})"
        )

        return df

    def fit(self, df: pd.DataFrame) -> dict:
        """Fit the distress prediction model with calibrated probabilities.

        Uses XGBoost with:
          - Class weighting for imbalanced data
          - Stratified cross-validation
          - Platt scaling for probability calibration
        """
        # Select available features
        available = [f for f in self.DISTRESS_FEATURES if f in df.columns]
        if len(available) < 3:
            raise ValueError(f"Need at least 3 features, found: {available}")

        self.feature_names = available
        target = "is_distressed"

        if target not in df.columns:
            raise ValueError("is_distressed column not found. Run create_distress_labels() first.")

        # Prepare data
        X = df[available].copy()
        for col in X.columns:
            X[col] = pd.to_numeric(X[col], errors="coerce")
        y = df[target]

        # Drop rows with NaN target
        mask = y.notna()
        X = X[mask]
        y = y[mask].astype(int)

        # Class balance
        pos_rate = y.mean()
        scale_pos_weight = (1 - pos_rate) / max(pos_rate, 0.01)

        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.params["test_size"],
            random_state=self.params["random_state"],
            stratify=y,
        )

        # Build pipeline
        self.pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", XGBClassifier(
                n_estimators=self.params["n_estimators"],
                max_depth=self.params["max_depth"],
                learning_rate=self.params["learning_rate"],
                scale_pos_weight=scale_pos_weight,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=self.params["random_state"],
                eval_metric="logloss",
                use_label_encoder=False,
            )),
        ])

        # Fit
        self.pipeline.fit(X_train, y_train)

        # Calibrate probabilities
        self.calibrated_model = CalibratedClassifierCV(
            self.pipeline, cv=3, method="sigmoid"
        )
        self.calibrated_model.fit(X_train, y_train)

        # Evaluate
        y_prob_test = self.calibrated_model.predict_proba(X_test)[:, 1]
        y_pred_test = (y_prob_test >= 0.5).astype(int)

        # Metrics
        roc_auc = roc_auc_score(y_test, y_prob_test)
        avg_precision = average_precision_score(y_test, y_prob_test)
        f1 = f1_score(y_test, y_pred_test)

        # Feature importance
        xgb_model = self.pipeline.named_steps["classifier"]
        importances = dict(zip(self.feature_names, xgb_model.feature_importances_))
        sorted_importance = dict(sorted(importances.items(), key=lambda x: x[1], reverse=True))

        # ROC curve
        fpr, tpr, thresholds = roc_curve(y_test, y_prob_test)

        # Precision-Recall curve
        precision, recall, pr_thresholds = precision_recall_curve(y_test, y_prob_test)

        self.results = {
            "roc_auc": roc_auc,
            "avg_precision": avg_precision,
            "f1_score": f1,
            "classification_report": classification_report(y_test, y_pred_test, output_dict=True),
            "feature_importance": sorted_importance,
            "n_train": len(X_train),
            "n_test": len(X_test),
            "pos_rate_train": y_train.mean(),
            "pos_rate_test": y_test.mean(),
            "roc_curve": {"fpr": fpr, "tpr": tpr, "thresholds": thresholds},
            "pr_curve": {"precision": precision, "recall": recall, "thresholds": pr_thresholds},
        }

        logger.info(
            f"Distress Predictor: ROC-AUC={roc_auc:.3f}, "
            f"Avg Precision={avg_precision:.3f}, F1={f1:.3f}"
        )
        logger.info(f"Top 5 risk factors: {list(sorted_importance.keys())[:5]}")

        return self.results

    def predict_distress_probability(self, df: pd.DataFrame) -> pd.DataFrame:
        """Score properties with distress probabilities.

        Returns DataFrame with:
          - distress_probability: Calibrated probability [0, 1]
          - risk_tier: Categorical risk classification
        """
        if self.calibrated_model is None:
            raise ValueError("Model not fitted. Call fit() first.")

        X = df[self.feature_names].copy()
        for col in X.columns:
            X[col] = pd.to_numeric(X[col], errors="coerce")

        probabilities = self.calibrated_model.predict_proba(X)[:, 1]

        result = df.copy()
        result["distress_probability"] = probabilities
        result["risk_tier"] = pd.cut(
            probabilities,
            bins=[0, 0.15, 0.35, 0.60, 1.0],
            labels=["Low Risk", "Moderate Risk", "Elevated Risk", "High Risk"],
        )

        logger.info(
            f"Scored {len(result)} properties. "
            f"Risk distribution: {result['risk_tier'].value_counts().to_dict()}"
        )

        return result

    def save(self, path=None):
        """Save model artifacts."""
        path = path or MODEL_ARTIFACTS / "distress_predictor.joblib"
        joblib.dump({
            "pipeline": self.pipeline,
            "calibrated_model": self.calibrated_model,
            "feature_names": self.feature_names,
            "results": self.results,
        }, path)
        logger.info(f"Distress predictor saved to {path}")
