"""
Hedonic Regression Model for Property Valuation
=================================================
Implements hedonic pricing theory to understand how individual property
features (location, size, age, building class, etc.) influence commercial
real estate values in NYC.

The hedonic approach decomposes property value into constituent characteristics,
enabling:
  - Fair value estimation for investment underwriting
  - Identification of mispriced assets (alpha generation)
  - Understanding of value drivers across sub-markets

Models:
  1. OLS Hedonic Regression (interpretability)
  2. Ridge/Lasso Regression (regularization for high-dimensional features)
  3. Gradient Boosted Regression (non-linear relationships)

Author: Allen Xu
"""

import numpy as np
import pandas as pd
import joblib
from loguru import logger
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    mean_absolute_percentage_error,
)
from sklearn.pipeline import Pipeline
import statsmodels.api as sm

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent.parent))
from config import MODEL_PARAMS, MODEL_ARTIFACTS


class HedonicRegressionModel:
    """Hedonic regression for NYC commercial property valuation.

    Predicts log(sale_price) from property characteristics, location,
    and market context features. Uses log-linear specification to
    capture the multiplicative nature of hedonic pricing.
    """

    # Features grouped by economic interpretation
    STRUCTURAL_FEATURES = [
        "gross_square_feet", "land_square_feet", "numfloors",
        "building_age", "floor_area_ratio", "commercial_ratio",
        "is_recently_renovated", "unitstotal",
    ]

    LOCATION_FEATURES = [
        "dist_to_cbd_km", "dist_to_grand_central_km",
        "dist_to_hudson_yards_km", "dist_to_world_trade_center_km",
        "min_landmark_dist_km", "is_manhattan",
    ]

    MARKET_FEATURES = [
        "neighborhood_avg_price", "neighborhood_sales_volume",
        "neighborhood_permit_activity", "price_vs_neighborhood",
    ]

    TEMPORAL_FEATURES = [
        "sale_year", "sale_quarter",
    ]

    def __init__(self, params: dict = None):
        self.params = params or MODEL_PARAMS["hedonic_regression"]
        self.pipeline = None
        self.ols_model = None
        self.feature_names = []
        self.results = {}

    def _select_features(self, df: pd.DataFrame) -> list[str]:
        """Select available features from the DataFrame."""
        all_features = (
            self.STRUCTURAL_FEATURES
            + self.LOCATION_FEATURES
            + self.MARKET_FEATURES
            + self.TEMPORAL_FEATURES
        )
        # Add any one-hot encoded columns (borough dummies, etc.)
        encoded_cols = [c for c in df.columns if any(
            c.startswith(prefix) for prefix in [
                "borough_name_", "building_class_category_encoded",
                "height_category_", "neighborhood_target_encoded",
            ]
        )]
        all_features.extend(encoded_cols)

        available = [f for f in all_features if f in df.columns]
        logger.info(f"Selected {len(available)} features for hedonic regression")
        return available

    def _prepare_data(self, df: pd.DataFrame):
        """Prepare feature matrix and target variable."""
        # Target: log of sale price (log-linear hedonic specification)
        if "sale_price" not in df.columns:
            raise ValueError("sale_price column required")

        target = np.log1p(df["sale_price"].astype(float))
        features = self._select_features(df)
        self.feature_names = features

        X = df[features].copy()

        # Ensure numeric
        for col in X.columns:
            X[col] = pd.to_numeric(X[col], errors="coerce")

        return X, target

    def fit_ols(self, df: pd.DataFrame) -> dict:
        """Fit OLS hedonic regression with statsmodels for interpretability.

        Returns detailed statistical results including:
          - Coefficient estimates and p-values
          - R-squared and adjusted R-squared
          - F-statistic
          - Heteroskedasticity-robust standard errors (HC3)
        """
        X, y = self._prepare_data(df)

        # Impute and add constant
        imputer = SimpleImputer(strategy="median")
        X_imputed = pd.DataFrame(
            imputer.fit_transform(X), columns=X.columns, index=X.index
        )
        X_const = sm.add_constant(X_imputed)

        # Drop rows with NaN in target
        mask = y.notna()
        X_const = X_const[mask]
        y_clean = y[mask]

        # Fit with heteroskedasticity-robust standard errors
        self.ols_model = sm.OLS(y_clean, X_const).fit(cov_type="HC3")

        # Extract results
        self.results["ols"] = {
            "r_squared": self.ols_model.rsquared,
            "adj_r_squared": self.ols_model.rsquared_adj,
            "f_statistic": self.ols_model.fvalue,
            "f_pvalue": self.ols_model.f_pvalue,
            "n_observations": int(self.ols_model.nobs),
            "coefficients": self.ols_model.params.to_dict(),
            "pvalues": self.ols_model.pvalues.to_dict(),
            "conf_int": self.ols_model.conf_int().to_dict(),
        }

        logger.info(
            f"OLS Hedonic Regression: R²={self.ols_model.rsquared:.4f}, "
            f"Adj-R²={self.ols_model.rsquared_adj:.4f}, "
            f"N={int(self.ols_model.nobs)}"
        )

        return self.results["ols"]

    def fit_regularized(self, df: pd.DataFrame, model_type: str = "ridge") -> dict:
        """Fit regularized hedonic regression with cross-validation.

        Args:
            df: Feature DataFrame with sale_price column
            model_type: 'ridge', 'lasso', or 'elasticnet'

        Returns:
            Dict with model performance metrics
        """
        X, y = self._prepare_data(df)

        # Split data
        mask = y.notna() & X.notna().all(axis=1)
        X_clean = X[mask]
        y_clean = y[mask]

        X_train, X_test, y_train, y_test = train_test_split(
            X_clean, y_clean,
            test_size=self.params["test_size"],
            random_state=self.params["random_state"],
        )

        # Build pipeline with imputation + scaling + regression
        estimators = {
            "ridge": Ridge(),
            "lasso": Lasso(max_iter=10000),
            "elasticnet": ElasticNet(max_iter=10000),
        }

        self.pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("regressor", estimators[model_type]),
        ])

        # Grid search for regularization strength
        param_grid = {
            "regressor__alpha": self.params["alpha_range"],
        }
        if model_type == "elasticnet":
            param_grid["regressor__l1_ratio"] = [0.1, 0.3, 0.5, 0.7, 0.9]

        grid_search = GridSearchCV(
            self.pipeline,
            param_grid,
            cv=self.params["cv_folds"],
            scoring="r2",
            n_jobs=-1,
        )
        grid_search.fit(X_train, y_train)

        self.pipeline = grid_search.best_estimator_

        # Evaluate
        y_pred_train = self.pipeline.predict(X_train)
        y_pred_test = self.pipeline.predict(X_test)

        self.results[model_type] = {
            "best_alpha": grid_search.best_params_["regressor__alpha"],
            "train_r2": r2_score(y_train, y_pred_train),
            "test_r2": r2_score(y_test, y_pred_test),
            "test_rmse": np.sqrt(mean_squared_error(y_test, y_pred_test)),
            "test_mae": mean_absolute_error(y_test, y_pred_test),
            "test_mape": mean_absolute_percentage_error(
                np.expm1(y_test), np.expm1(y_pred_test)
            ),
            "cv_r2_mean": grid_search.best_score_,
            "n_train": len(X_train),
            "n_test": len(X_test),
        }

        logger.info(
            f"{model_type.upper()} Regression: "
            f"Test R²={self.results[model_type]['test_r2']:.4f}, "
            f"MAPE={self.results[model_type]['test_mape']:.1%}"
        )

        return self.results[model_type]

    def fit_gradient_boosting(self, df: pd.DataFrame) -> dict:
        """Fit gradient boosted regression for capturing non-linear effects.

        Gradient boosting captures:
          - Non-linear price-feature relationships
          - Feature interactions (e.g., location × size)
          - Heterogeneous effects across sub-markets
        """
        X, y = self._prepare_data(df)

        mask = y.notna() & X.notna().all(axis=1)
        X_clean = X[mask]
        y_clean = y[mask]

        X_train, X_test, y_train, y_test = train_test_split(
            X_clean, y_clean,
            test_size=self.params["test_size"],
            random_state=self.params["random_state"],
        )

        gb_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("regressor", GradientBoostingRegressor(
                n_estimators=500,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                random_state=self.params["random_state"],
            )),
        ])

        gb_pipeline.fit(X_train, y_train)

        y_pred_train = gb_pipeline.predict(X_train)
        y_pred_test = gb_pipeline.predict(X_test)

        # Feature importance
        importances = gb_pipeline.named_steps["regressor"].feature_importances_
        feature_importance = dict(zip(self.feature_names, importances))
        sorted_importance = dict(
            sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        )

        self.results["gradient_boosting"] = {
            "train_r2": r2_score(y_train, y_pred_train),
            "test_r2": r2_score(y_test, y_pred_test),
            "test_rmse": np.sqrt(mean_squared_error(y_test, y_pred_test)),
            "test_mae": mean_absolute_error(y_test, y_pred_test),
            "test_mape": mean_absolute_percentage_error(
                np.expm1(y_test), np.expm1(y_pred_test)
            ),
            "feature_importance": sorted_importance,
            "n_train": len(X_train),
            "n_test": len(X_test),
        }

        # Store for predictions
        self.gb_pipeline = gb_pipeline

        logger.info(
            f"Gradient Boosting: Test R²={self.results['gradient_boosting']['test_r2']:.4f}, "
            f"MAPE={self.results['gradient_boosting']['test_mape']:.1%}"
        )
        logger.info(f"Top 5 features: {list(sorted_importance.keys())[:5]}")

        return self.results["gradient_boosting"]

    def predict_value(self, property_features: pd.DataFrame) -> pd.Series:
        """Predict property values using the best fitted model.

        Returns predictions in actual dollar values (not log scale).
        """
        if self.pipeline is None and not hasattr(self, "gb_pipeline"):
            raise ValueError("No model fitted. Call fit_regularized() or fit_gradient_boosting() first.")

        model = getattr(self, "gb_pipeline", self.pipeline)
        X = property_features[self.feature_names].copy()
        for col in X.columns:
            X[col] = pd.to_numeric(X[col], errors="coerce")

        log_predictions = model.predict(X)
        return pd.Series(np.expm1(log_predictions), index=X.index, name="predicted_value")

    def get_coefficient_summary(self) -> pd.DataFrame:
        """Get a formatted summary of OLS hedonic coefficients.

        Interpretation: Since the model uses log(price), coefficients
        represent the percentage change in price for a one-unit change
        in the feature (semi-elasticity).
        """
        if self.ols_model is None:
            raise ValueError("OLS model not fitted. Call fit_ols() first.")

        summary = pd.DataFrame({
            "coefficient": self.ols_model.params,
            "std_error": self.ols_model.bse,
            "t_statistic": self.ols_model.tvalues,
            "p_value": self.ols_model.pvalues,
            "ci_lower": self.ols_model.conf_int()[0],
            "ci_upper": self.ols_model.conf_int()[1],
            "pct_impact": (np.exp(self.ols_model.params) - 1) * 100,
        })

        return summary.sort_values("p_value")

    def save(self, path=None):
        """Save model artifacts."""
        path = path or MODEL_ARTIFACTS / "hedonic_regression.joblib"
        joblib.dump({
            "pipeline": self.pipeline,
            "ols_model": self.ols_model,
            "feature_names": self.feature_names,
            "results": self.results,
        }, path)
        logger.info(f"Model saved to {path}")

    def load(self, path=None):
        """Load model artifacts."""
        path = path or MODEL_ARTIFACTS / "hedonic_regression.joblib"
        artifacts = joblib.load(path)
        self.pipeline = artifacts["pipeline"]
        self.ols_model = artifacts["ols_model"]
        self.feature_names = artifacts["feature_names"]
        self.results = artifacts["results"]
        logger.info(f"Model loaded from {path}")
