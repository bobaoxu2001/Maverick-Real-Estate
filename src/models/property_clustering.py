"""
Property & Market Clustering Model
====================================
Segments NYC commercial properties and neighborhoods into meaningful
clusters using unsupervised learning techniques.

Use Cases:
  - Identify comparable property groups for valuation
  - Discover sub-market segments with similar risk/return profiles
  - Group property owners by portfolio characteristics
  - Find emerging neighborhoods with similar growth patterns

Methods:
  1. K-Means Clustering (centroid-based, interpretable segments)
  2. DBSCAN (density-based, for detecting spatial clusters and outliers)
  3. Hierarchical Clustering (for dendrogram-based market taxonomy)

Author: Allen Xu
"""

import numpy as np
import pandas as pd
import joblib
from loguru import logger
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent.parent))
from config import MODEL_PARAMS, MODEL_ARTIFACTS


class PropertyClusteringModel:
    """Clustering analysis for NYC CRE market segmentation.

    Discovers natural groupings among properties and neighborhoods
    to identify sub-markets, comps, and investment themes.
    """

    PROPERTY_FEATURES = [
        "price_per_sqft", "gross_square_feet", "building_age",
        "numfloors", "floor_area_ratio", "commercial_ratio",
        "dist_to_cbd_km", "assesstot",
    ]

    NEIGHBORHOOD_FEATURES = [
        "neighborhood_avg_price", "neighborhood_sales_volume",
        "neighborhood_median_price", "neighborhood_price_std",
    ]

    def __init__(self, params: dict = None):
        self.params = params or MODEL_PARAMS["clustering"]
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy="median")
        self.kmeans_model = None
        self.dbscan_model = None
        self.pca = None
        self.results = {}

    def _prepare_features(
        self, df: pd.DataFrame, feature_set: str = "property"
    ) -> tuple[np.ndarray, list[str], pd.Index]:
        """Prepare and scale features for clustering."""
        if feature_set == "property":
            features = self.PROPERTY_FEATURES
        else:
            features = self.PROPERTY_FEATURES + self.NEIGHBORHOOD_FEATURES

        available = [f for f in features if f in df.columns]
        if len(available) < 2:
            raise ValueError(f"Need at least 2 features, found {len(available)}: {available}")

        X = df[available].copy()
        for col in X.columns:
            X[col] = pd.to_numeric(X[col], errors="coerce")

        # Track valid indices
        valid_mask = X.notna().all(axis=1)
        X_valid = X[valid_mask]

        X_imputed = self.imputer.fit_transform(X_valid)
        X_scaled = self.scaler.fit_transform(X_imputed)

        logger.info(f"Prepared {X_scaled.shape[0]} samples × {X_scaled.shape[1]} features")
        return X_scaled, available, X_valid.index

    def find_optimal_k(
        self, df: pd.DataFrame, max_k: int = None, feature_set: str = "property"
    ) -> dict:
        """Find optimal number of clusters using elbow method and silhouette analysis.

        Returns dict with:
          - inertias: Sum of squared distances for each k
          - silhouette_scores: Silhouette score for each k
          - optimal_k: Recommended number of clusters
        """
        max_k = max_k or self.params["max_clusters"]
        X, features, idx = self._prepare_features(df, feature_set)

        inertias = []
        silhouette_scores = []
        calinski_scores = []
        k_range = range(2, min(max_k + 1, len(X)))

        for k in k_range:
            km = KMeans(
                n_clusters=k,
                random_state=self.params["random_state"],
                n_init=10,
            )
            labels = km.fit_predict(X)
            inertias.append(km.inertia_)
            silhouette_scores.append(silhouette_score(X, labels))
            calinski_scores.append(calinski_harabasz_score(X, labels))

        # Optimal k: highest silhouette score
        optimal_k = list(k_range)[np.argmax(silhouette_scores)]

        self.results["optimal_k"] = {
            "k_range": list(k_range),
            "inertias": inertias,
            "silhouette_scores": silhouette_scores,
            "calinski_scores": calinski_scores,
            "optimal_k": optimal_k,
            "best_silhouette": max(silhouette_scores),
        }

        logger.info(
            f"Optimal K={optimal_k} (silhouette={max(silhouette_scores):.3f})"
        )
        return self.results["optimal_k"]

    def fit_kmeans(
        self, df: pd.DataFrame, n_clusters: int = None, feature_set: str = "property"
    ) -> pd.DataFrame:
        """Fit K-Means clustering and return DataFrame with cluster assignments.

        Each cluster represents a distinct market segment with
        interpretable centroid characteristics.
        """
        X, features, idx = self._prepare_features(df, feature_set)

        if n_clusters is None:
            if "optimal_k" in self.results:
                n_clusters = self.results["optimal_k"]["optimal_k"]
            else:
                n_clusters = 5

        self.kmeans_model = KMeans(
            n_clusters=n_clusters,
            random_state=self.params["random_state"],
            n_init=10,
        )
        labels = self.kmeans_model.fit_predict(X)

        # Create result DataFrame
        result = df.loc[idx].copy()
        result["cluster"] = labels

        # Cluster profiles (centroids in original scale)
        centroids_scaled = self.kmeans_model.cluster_centers_
        centroids_original = self.scaler.inverse_transform(centroids_scaled)
        centroid_df = pd.DataFrame(centroids_original, columns=features)
        centroid_df.index.name = "cluster"

        # Cluster statistics
        cluster_stats = result.groupby("cluster").agg({
            features[0]: ["mean", "count"],
        })
        cluster_stats.columns = ["mean_primary_feature", "count"]

        self.results["kmeans"] = {
            "n_clusters": n_clusters,
            "silhouette_score": silhouette_score(X, labels),
            "inertia": self.kmeans_model.inertia_,
            "centroids": centroid_df,
            "cluster_sizes": pd.Series(labels).value_counts().to_dict(),
        }

        logger.info(
            f"K-Means: {n_clusters} clusters, "
            f"silhouette={self.results['kmeans']['silhouette_score']:.3f}"
        )

        return result

    def fit_dbscan(
        self,
        df: pd.DataFrame,
        eps: float = 0.5,
        min_samples: int = None,
        feature_set: str = "property",
    ) -> pd.DataFrame:
        """Fit DBSCAN for density-based clustering.

        Advantages over K-Means:
          - Automatically discovers number of clusters
          - Identifies outlier properties (noise points)
          - Finds arbitrarily shaped clusters (e.g., geographic corridors)
        """
        min_samples = min_samples or self.params["min_samples_dbscan"]
        X, features, idx = self._prepare_features(df, feature_set)

        self.dbscan_model = DBSCAN(eps=eps, min_samples=min_samples)
        labels = self.dbscan_model.fit_predict(X)

        result = df.loc[idx].copy()
        result["cluster"] = labels

        n_clusters = len(set(labels) - {-1})
        n_noise = (labels == -1).sum()

        self.results["dbscan"] = {
            "n_clusters": n_clusters,
            "n_noise_points": int(n_noise),
            "noise_ratio": n_noise / len(labels),
            "cluster_sizes": pd.Series(labels[labels >= 0]).value_counts().to_dict(),
        }

        if n_clusters > 1:
            non_noise_mask = labels >= 0
            self.results["dbscan"]["silhouette_score"] = silhouette_score(
                X[non_noise_mask], labels[non_noise_mask]
            )

        logger.info(
            f"DBSCAN: {n_clusters} clusters, {n_noise} noise points "
            f"({n_noise/len(labels):.1%})"
        )

        return result

    def get_cluster_profiles(self, clustered_df: pd.DataFrame) -> pd.DataFrame:
        """Generate interpretable cluster profiles.

        For each cluster, computes:
          - Mean and median of key metrics
          - Dominant building type
          - Geographic centroid
          - Investment characterization
        """
        profiles = []

        for cluster_id in sorted(clustered_df["cluster"].unique()):
            if cluster_id == -1:
                continue  # Skip noise points from DBSCAN

            mask = clustered_df["cluster"] == cluster_id
            subset = clustered_df[mask]

            profile = {"cluster": cluster_id, "count": len(subset)}

            # Numeric summaries
            for col in self.PROPERTY_FEATURES:
                if col in subset.columns:
                    vals = pd.to_numeric(subset[col], errors="coerce")
                    profile[f"{col}_mean"] = vals.mean()
                    profile[f"{col}_median"] = vals.median()

            # Geographic centroid
            if "latitude" in subset.columns and "longitude" in subset.columns:
                profile["centroid_lat"] = pd.to_numeric(subset["latitude"], errors="coerce").mean()
                profile["centroid_lon"] = pd.to_numeric(subset["longitude"], errors="coerce").mean()

            # Dominant borough
            if "borough_name" in subset.columns:
                profile["dominant_borough"] = subset["borough_name"].mode().iloc[0] if len(subset["borough_name"].mode()) > 0 else "Unknown"

            profiles.append(profile)

        profile_df = pd.DataFrame(profiles)
        logger.info(f"Generated profiles for {len(profile_df)} clusters")
        return profile_df

    def reduce_for_visualization(
        self, df: pd.DataFrame, n_components: int = 2, feature_set: str = "property"
    ) -> pd.DataFrame:
        """Apply PCA for 2D/3D cluster visualization."""
        X, features, idx = self._prepare_features(df, feature_set)

        self.pca = PCA(n_components=n_components)
        X_reduced = self.pca.fit_transform(X)

        result = df.loc[idx].copy()
        for i in range(n_components):
            result[f"pca_{i+1}"] = X_reduced[:, i]

        logger.info(
            f"PCA reduction: {X.shape[1]}D → {n_components}D "
            f"(variance explained: {self.pca.explained_variance_ratio_.sum():.1%})"
        )
        return result

    def save(self, path=None):
        """Save clustering model artifacts."""
        path = path or MODEL_ARTIFACTS / "property_clustering.joblib"
        joblib.dump({
            "kmeans_model": self.kmeans_model,
            "dbscan_model": self.dbscan_model,
            "scaler": self.scaler,
            "imputer": self.imputer,
            "pca": self.pca,
            "results": self.results,
        }, path)
        logger.info(f"Clustering model saved to {path}")
