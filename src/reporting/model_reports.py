"""
Automated model reporting for professional analytics deliverables.

Generates:
  - Model metrics table (CSV)
  - JSON artifact with serializable model outputs
  - Stakeholder-ready PNG charts
  - HTML report stitching metrics + figures
"""

from __future__ import annotations

import json
from datetime import datetime, date
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from loguru import logger

import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent.parent))
from config import REPORTS_MODEL_DIR


class ModelReportGenerator:
    """Generate model performance artifacts for executive and analyst audiences."""

    def __init__(self, output_root: Path | None = None):
        self.output_root = Path(output_root) if output_root else REPORTS_MODEL_DIR
        self.output_root.mkdir(parents=True, exist_ok=True)

    def generate(
        self,
        model_results: dict[str, Any],
        featured_df: pd.DataFrame | None = None,
    ) -> dict[str, str]:
        """Generate report bundle and return saved artifact paths."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = self.output_root / timestamp
        figures_dir = run_dir / "figures"
        run_dir.mkdir(parents=True, exist_ok=True)
        figures_dir.mkdir(parents=True, exist_ok=True)

        metrics_df = self._build_metrics_table(model_results)
        metrics_path = run_dir / "model_metrics.csv"
        metrics_df.to_csv(metrics_path, index=False)

        json_path = run_dir / "model_results.json"
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(self._to_json_safe(model_results), f, indent=2)

        image_paths = []
        leaderboard = self._plot_model_leaderboard(metrics_df, figures_dir)
        if leaderboard:
            image_paths.append(leaderboard)

        for model_key, output_name, title in [
            ("hedonic_gb", "hedonic_gb_feature_importance.png", "Hedonic GB Feature Importance"),
            ("hedonic_stacked", "hedonic_stacked_feature_importance.png", "Hedonic Stacked Feature Importance"),
            ("distress", "distress_feature_importance.png", "Distress Predictor Feature Importance"),
        ]:
            fig_path = self._plot_feature_importance(model_results, model_key, figures_dir / output_name, title)
            if fig_path:
                image_paths.append(fig_path)

        if featured_df is not None and not featured_df.empty:
            market_box = self._plot_market_distribution(featured_df, figures_dir / "market_distribution.png")
            if market_box:
                image_paths.append(market_box)

            market_scatter = self._plot_market_scatter(featured_df, figures_dir / "market_scatter.png")
            if market_scatter:
                image_paths.append(market_scatter)

        html_path = self._write_html_report(run_dir, metrics_df, image_paths)

        logger.info(f"Model report generated: {run_dir}")
        return {
            "run_dir": str(run_dir),
            "metrics_csv": str(metrics_path),
            "model_results_json": str(json_path),
            "html_report": str(html_path),
            "n_figures": str(len(image_paths)),
        }

    @staticmethod
    def _build_metrics_table(model_results: dict[str, Any]) -> pd.DataFrame:
        rows = []
        for model_name, payload in model_results.items():
            if not isinstance(payload, dict):
                continue
            row = {
                "model": model_name,
                "r2": payload.get("test_r2", payload.get("r_squared")),
                "rmse": payload.get("test_rmse", payload.get("rmse")),
                "mae": payload.get("test_mae", payload.get("mae")),
                "mape": payload.get("test_mape", payload.get("mape")),
                "roc_auc": payload.get("roc_auc"),
                "avg_precision": payload.get("avg_precision"),
                "f1_score": payload.get("f1_score"),
                "n_train": payload.get("n_train"),
                "n_test": payload.get("n_test"),
            }
            rows.append(row)

        metrics_df = pd.DataFrame(rows)
        if metrics_df.empty:
            return pd.DataFrame(columns=["model", "r2", "rmse", "mae", "mape", "roc_auc", "avg_precision", "f1_score", "n_train", "n_test"])

        return metrics_df.sort_values(by=["r2", "roc_auc"], ascending=False, na_position="last").reset_index(drop=True)

    @staticmethod
    def _plot_model_leaderboard(metrics_df: pd.DataFrame, figures_dir: Path) -> Path | None:
        if metrics_df.empty or metrics_df["r2"].dropna().empty:
            return None

        plot_df = metrics_df.dropna(subset=["r2"]).copy()
        plot_df = plot_df.sort_values("r2", ascending=True)
        if plot_df.empty:
            return None

        plt.figure(figsize=(10, 6))
        sns.barplot(data=plot_df, x="r2", y="model", palette="Blues_r")
        plt.title("Model Leaderboard (R²)")
        plt.xlabel("Out-of-sample R²")
        plt.ylabel("Model")
        plt.tight_layout()
        out_path = figures_dir / "model_leaderboard_r2.png"
        plt.savefig(out_path, dpi=180)
        plt.close()
        return out_path

    @staticmethod
    def _plot_feature_importance(
        model_results: dict[str, Any],
        model_key: str,
        out_path: Path,
        title: str,
        top_n: int = 15,
    ) -> Path | None:
        payload = model_results.get(model_key, {})
        if not isinstance(payload, dict):
            return None
        feature_importance = payload.get("feature_importance")
        if not isinstance(feature_importance, dict) or not feature_importance:
            return None

        fi_df = pd.DataFrame(
            {"feature": list(feature_importance.keys()), "importance": list(feature_importance.values())}
        )
        fi_df = fi_df.sort_values("importance", ascending=False).head(top_n).sort_values("importance")

        plt.figure(figsize=(11, 7))
        sns.barplot(data=fi_df, x="importance", y="feature", palette="viridis")
        plt.title(title)
        plt.xlabel("Importance")
        plt.ylabel("")
        plt.tight_layout()
        plt.savefig(out_path, dpi=180)
        plt.close()
        return out_path

    @staticmethod
    def _plot_market_distribution(df: pd.DataFrame, out_path: Path) -> Path | None:
        if "borough_name" not in df.columns or "price_per_sqft" not in df.columns:
            return None

        plot_df = df[["borough_name", "price_per_sqft"]].copy()
        plot_df["price_per_sqft"] = pd.to_numeric(plot_df["price_per_sqft"], errors="coerce")
        plot_df = plot_df.dropna()
        if plot_df.empty:
            return None

        plot_df = plot_df[plot_df["price_per_sqft"] > 0]
        if plot_df.empty:
            return None

        plt.figure(figsize=(11, 6))
        sns.boxplot(data=plot_df, x="borough_name", y="price_per_sqft", palette="Set2")
        plt.yscale("log")
        plt.title("Price per SF Distribution by Borough (log scale)")
        plt.xlabel("")
        plt.ylabel("Price per SF (log)")
        plt.xticks(rotation=15)
        plt.tight_layout()
        plt.savefig(out_path, dpi=180)
        plt.close()
        return out_path

    @staticmethod
    def _plot_market_scatter(df: pd.DataFrame, out_path: Path) -> Path | None:
        needed = {"dist_to_cbd_km", "price_per_sqft"}
        if not needed.issubset(set(df.columns)):
            return None

        scatter_cols = ["dist_to_cbd_km", "price_per_sqft"]
        if "borough_name" in df.columns:
            scatter_cols.append("borough_name")

        plot_df = df[scatter_cols].copy()
        plot_df["dist_to_cbd_km"] = pd.to_numeric(plot_df["dist_to_cbd_km"], errors="coerce")
        plot_df["price_per_sqft"] = pd.to_numeric(plot_df["price_per_sqft"], errors="coerce")
        plot_df = plot_df.dropna(subset=["dist_to_cbd_km", "price_per_sqft"])
        if plot_df.empty:
            return None

        plot_df = plot_df.sample(n=min(len(plot_df), 4000), random_state=42)

        plt.figure(figsize=(11, 6))
        if "borough_name" in plot_df.columns:
            sns.scatterplot(
                data=plot_df,
                x="dist_to_cbd_km",
                y="price_per_sqft",
                hue="borough_name",
                alpha=0.5,
                s=20,
            )
            plt.legend(title="Borough", bbox_to_anchor=(1.02, 1), loc="upper left")
        else:
            sns.scatterplot(data=plot_df, x="dist_to_cbd_km", y="price_per_sqft", alpha=0.5, s=20)

        plt.yscale("log")
        plt.title("Distance to CBD vs Price per SF")
        plt.xlabel("Distance to CBD (km)")
        plt.ylabel("Price per SF (log)")
        plt.tight_layout()
        plt.savefig(out_path, dpi=180)
        plt.close()
        return out_path

    @staticmethod
    def _write_html_report(run_dir: Path, metrics_df: pd.DataFrame, image_paths: list[Path]) -> Path:
        report_path = run_dir / "report.html"
        table_html = metrics_df.to_html(index=False, float_format=lambda x: f"{x:,.4f}")
        images_html = "\n".join(
            [
                (
                    f"<div class='chart'><h3>{img.stem.replace('_', ' ').title()}</h3>"
                    f"<img src='figures/{img.name}' alt='{img.name}'/></div>"
                )
                for img in image_paths
            ]
        )

        html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>NYC CRE Model Performance Report</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; color: #1f2937; }}
    h1 {{ margin-bottom: 4px; }}
    .subtitle {{ color: #6b7280; margin-bottom: 24px; }}
    table {{ border-collapse: collapse; width: 100%; margin-bottom: 24px; }}
    th, td {{ border: 1px solid #d1d5db; padding: 8px; text-align: left; }}
    th {{ background: #f3f4f6; }}
    .chart {{ margin: 18px 0 28px 0; }}
    .chart img {{ max-width: 100%; border: 1px solid #e5e7eb; border-radius: 8px; }}
  </style>
</head>
<body>
  <h1>NYC CRE Investment Intelligence — Model Report</h1>
  <div class="subtitle">Author: Allen Xu | Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</div>
  <h2>Model Metrics</h2>
  {table_html}
  <h2>Diagnostics & Visuals</h2>
  {images_html}
</body>
</html>
"""
        report_path.write_text(html, encoding="utf-8")
        return report_path

    def _to_json_safe(self, value: Any) -> Any:
        if isinstance(value, dict):
            return {str(k): self._to_json_safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [self._to_json_safe(v) for v in value]
        if isinstance(value, (np.integer, np.floating)):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, pd.Series):
            return self._to_json_safe(value.to_dict())
        if isinstance(value, pd.DataFrame):
            return self._to_json_safe(value.head(200).to_dict(orient="records"))
        if isinstance(value, (datetime, date, pd.Timestamp)):
            return value.isoformat()
        return value
