"""
Time Series Forecasting for CRE Market Trends
===============================================
Models temporal dynamics in NYC commercial real estate markets to
forecast price trends, identify market cycles, and inform investment
timing decisions.

Key Applications:
  - Forecasting price per sqft trends by neighborhood/borough
  - Identifying cyclical patterns in transaction volumes
  - Correlating CRE trends with macroeconomic indicators
  - Projecting market conditions for investment horizons

Models:
  1. ARIMA/SARIMAX — Classical time series with macro exogenous variables
  2. Prophet — Facebook's decomposable time series (trend + seasonality + holidays)
  3. Rolling Statistics — Trend and momentum indicators

Author: Allen Xu
"""

import numpy as np
import pandas as pd
import joblib
from loguru import logger
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, acf, pacf
from sklearn.metrics import mean_absolute_error, mean_squared_error

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent.parent))
from config import MODEL_PARAMS, MODEL_ARTIFACTS


class TimeSeriesForecaster:
    """Time series analysis and forecasting for NYC CRE markets.

    Analyzes and forecasts price trends, transaction volumes, and
    market indicators over multi-year horizons.
    """

    def __init__(self, params: dict = None):
        self.params = params or MODEL_PARAMS["time_series"]
        self.sarimax_model = None
        self.prophet_model = None
        self.results = {}

    # ── Data Preparation ──────────────────────────────

    @staticmethod
    def prepare_price_series(
        sales_df: pd.DataFrame,
        freq: str = "MS",
        metric: str = "median",
        group_by: str = None,
    ) -> pd.DataFrame:
        """Aggregate sales data into time series.

        Args:
            sales_df: Sales DataFrame with sale_date and sale_price
            freq: Resampling frequency ('MS' = month start, 'QS' = quarter start)
            metric: Aggregation method ('median', 'mean')
            group_by: Optional grouping column (e.g., 'borough_name')

        Returns:
            Time series DataFrame indexed by date
        """
        df = sales_df.copy()
        df["sale_date"] = pd.to_datetime(df["sale_date"], errors="coerce")
        df["sale_price"] = pd.to_numeric(df["sale_price"], errors="coerce")
        df = df.dropna(subset=["sale_date", "sale_price"])
        df = df.set_index("sale_date").sort_index()

        if "price_per_sqft" in df.columns:
            df["price_per_sqft"] = pd.to_numeric(df["price_per_sqft"], errors="coerce")

        if group_by and group_by in df.columns:
            # Create separate series for each group
            groups = df.groupby(group_by)
            series_dict = {}
            for name, group in groups:
                resampled = group.resample(freq).agg({
                    "sale_price": metric,
                    "price_per_sqft": metric if "price_per_sqft" in group.columns else "count",
                })
                resampled["transaction_volume"] = group.resample(freq)["sale_price"].count()
                series_dict[name] = resampled
            return series_dict
        else:
            result = df.resample(freq).agg({
                "sale_price": metric,
            })
            result["transaction_volume"] = df.resample(freq)["sale_price"].count()
            if "price_per_sqft" in df.columns:
                result["price_per_sqft"] = df.resample(freq)["price_per_sqft"].agg(metric)
            return result

    # ── Stationarity & Diagnostics ────────────────────

    @staticmethod
    def test_stationarity(series: pd.Series) -> dict:
        """Augmented Dickey-Fuller test for stationarity.

        Returns:
            Dict with ADF statistic, p-value, and stationarity assessment
        """
        series_clean = series.dropna()
        if len(series_clean) < 20:
            return {"error": "Insufficient data points"}

        result = adfuller(series_clean, autolag="AIC")

        diagnostics = {
            "adf_statistic": result[0],
            "p_value": result[1],
            "n_lags": result[2],
            "n_observations": result[3],
            "critical_values": result[4],
            "is_stationary": result[1] < 0.05,
        }

        logger.info(
            f"ADF Test: stat={result[0]:.4f}, p={result[1]:.4f} → "
            f"{'Stationary' if result[1] < 0.05 else 'Non-stationary'}"
        )
        return diagnostics

    @staticmethod
    def decompose_series(
        series: pd.Series, period: int = 12, model: str = "additive"
    ) -> dict:
        """Decompose time series into trend, seasonal, and residual components."""
        series_clean = series.dropna()
        if len(series_clean) < 2 * period:
            logger.warning("Insufficient data for decomposition")
            return {}

        decomposition = seasonal_decompose(series_clean, model=model, period=period)

        return {
            "trend": decomposition.trend,
            "seasonal": decomposition.seasonal,
            "residual": decomposition.resid,
            "observed": decomposition.observed,
        }

    # ── SARIMAX Model ─────────────────────────────────

    def fit_sarimax(
        self,
        series: pd.Series,
        order: tuple = (1, 1, 1),
        seasonal_order: tuple = (1, 1, 1, 12),
        exog: pd.DataFrame = None,
    ) -> dict:
        """Fit SARIMAX model with optional macroeconomic exogenous variables.

        The exogenous variables allow us to model how macro factors
        (interest rates, CPI, unemployment) influence CRE prices.

        Args:
            series: Target time series (e.g., median price)
            order: ARIMA order (p, d, q)
            seasonal_order: Seasonal order (P, D, Q, s)
            exog: Exogenous variables DataFrame (aligned index)
        """
        series_clean = series.dropna()

        if exog is not None:
            # Align exogenous variables with target series
            exog_aligned = exog.reindex(series_clean.index).ffill().bfill()
            exog_aligned = exog_aligned.select_dtypes(include=[np.number])
        else:
            exog_aligned = None

        # Split for validation (last 20% as test)
        split_idx = int(len(series_clean) * 0.8)
        train = series_clean[:split_idx]
        test = series_clean[split_idx:]

        exog_train = exog_aligned[:split_idx] if exog_aligned is not None else None
        exog_test = exog_aligned[split_idx:] if exog_aligned is not None else None

        # Fit model
        try:
            model = SARIMAX(
                train,
                order=order,
                seasonal_order=seasonal_order,
                exog=exog_train,
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            self.sarimax_model = model.fit(disp=False, maxiter=500)

            # In-sample diagnostics
            aic = self.sarimax_model.aic
            bic = self.sarimax_model.bic

            # Out-of-sample forecast
            forecast = self.sarimax_model.forecast(
                steps=len(test),
                exog=exog_test,
            )

            # Metrics
            mae = mean_absolute_error(test, forecast)
            rmse = np.sqrt(mean_squared_error(test, forecast))
            mape = np.mean(np.abs((test - forecast) / test)) * 100

            self.results["sarimax"] = {
                "order": order,
                "seasonal_order": seasonal_order,
                "aic": aic,
                "bic": bic,
                "mae": mae,
                "rmse": rmse,
                "mape": mape,
                "n_train": len(train),
                "n_test": len(test),
                "forecast_values": forecast,
                "actual_values": test,
                "has_exog": exog is not None,
            }

            logger.info(
                f"SARIMAX{order}×{seasonal_order}: "
                f"AIC={aic:.1f}, MAE={mae:.0f}, MAPE={mape:.1f}%"
            )

        except Exception as e:
            logger.error(f"SARIMAX fitting failed: {e}")
            self.results["sarimax"] = {"error": str(e)}

        return self.results.get("sarimax", {})

    def forecast_sarimax(
        self, periods: int = None, exog_future: pd.DataFrame = None
    ) -> pd.DataFrame:
        """Generate forward-looking forecasts with confidence intervals."""
        periods = periods or self.params["forecast_periods"]

        if self.sarimax_model is None:
            raise ValueError("SARIMAX model not fitted. Call fit_sarimax() first.")

        forecast = self.sarimax_model.get_forecast(steps=periods, exog=exog_future)
        forecast_df = pd.DataFrame({
            "forecast": forecast.predicted_mean,
            "ci_lower": forecast.conf_int().iloc[:, 0],
            "ci_upper": forecast.conf_int().iloc[:, 1],
        })

        logger.info(f"Generated {periods}-period forecast")
        return forecast_df

    # ── Prophet Model ─────────────────────────────────

    def fit_prophet(
        self, series: pd.Series, macro_df: pd.DataFrame = None
    ) -> dict:
        """Fit Facebook Prophet for trend + seasonality decomposition.

        Prophet advantages for CRE:
          - Handles missing data and outliers gracefully
          - Automatic changepoint detection (market regime shifts)
          - Intuitive decomposition for stakeholder communication
          - Easy incorporation of macro regressors
        """
        try:
            from prophet import Prophet
        except ImportError:
            logger.warning("Prophet not installed — skipping Prophet model")
            return {"error": "Prophet not installed"}

        # Prepare Prophet format
        prophet_df = pd.DataFrame({
            "ds": series.index,
            "y": series.values,
        }).dropna()

        if len(prophet_df) < 24:
            return {"error": "Insufficient data (need 24+ months)"}

        # Split
        split_idx = int(len(prophet_df) * 0.8)
        train = prophet_df[:split_idx]
        test = prophet_df[split_idx:]

        # Configure Prophet
        model = Prophet(
            changepoint_prior_scale=self.params["changepoint_prior_scale"],
            seasonality_mode="multiplicative",
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
        )

        # Add macro regressors if available
        if macro_df is not None:
            macro_aligned = macro_df.reindex(prophet_df["ds"]).ffill().bfill()
            for col in macro_aligned.select_dtypes(include=[np.number]).columns[:5]:
                model.add_regressor(col)
                train[col] = macro_aligned[col][:split_idx].values
                test[col] = macro_aligned[col][split_idx:].values

        model.fit(train)

        # Forecast
        forecast = model.predict(test)

        mae = mean_absolute_error(test["y"].values, forecast["yhat"].values[:len(test)])
        mape = np.mean(np.abs((test["y"].values - forecast["yhat"].values[:len(test)]) / test["y"].values)) * 100

        self.prophet_model = model
        self.results["prophet"] = {
            "mae": mae,
            "mape": mape,
            "n_train": len(train),
            "n_test": len(test),
            "changepoints": model.changepoints.tolist() if hasattr(model, "changepoints") else [],
        }

        logger.info(f"Prophet: MAE={mae:.0f}, MAPE={mape:.1f}%")
        return self.results["prophet"]

    def forecast_prophet(self, periods: int = None) -> pd.DataFrame:
        """Generate Prophet forecast."""
        periods = periods or self.params["forecast_periods"]

        if self.prophet_model is None:
            raise ValueError("Prophet model not fitted")

        future = self.prophet_model.make_future_dataframe(periods=periods, freq="MS")
        forecast = self.prophet_model.predict(future)

        return forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(periods)

    # ── Rolling Statistics ────────────────────────────

    @staticmethod
    def compute_momentum_indicators(series: pd.Series) -> pd.DataFrame:
        """Compute rolling statistics and momentum indicators.

        Useful for identifying trend direction and strength:
          - Moving averages (6-month, 12-month)
          - Rate of change
          - Volatility (rolling std)
          - Z-score (deviation from trend)
        """
        df = pd.DataFrame({"value": series})

        # Moving averages
        df["ma_6m"] = series.rolling(6, min_periods=3).mean()
        df["ma_12m"] = series.rolling(12, min_periods=6).mean()

        # Rate of change (YoY)
        df["yoy_change"] = series.pct_change(12)

        # Momentum (6-month)
        df["momentum_6m"] = series.pct_change(6)

        # Volatility
        df["volatility_12m"] = series.rolling(12, min_periods=6).std()

        # Z-score (how far from 12-month moving average)
        df["z_score"] = (series - df["ma_12m"]) / df["volatility_12m"]

        # Trend signal
        df["trend_signal"] = np.where(
            df["ma_6m"] > df["ma_12m"], "bullish", "bearish"
        )

        return df

    def save(self, path=None):
        """Save time series model artifacts."""
        path = path or MODEL_ARTIFACTS / "time_series_forecast.joblib"
        joblib.dump({
            "sarimax_model": self.sarimax_model,
            "results": self.results,
        }, path)
        logger.info(f"Time series models saved to {path}")
