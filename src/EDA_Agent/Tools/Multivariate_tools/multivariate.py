#====================================================
# Multivariate Statistics
#====================================================
"""
Multivariate Exploratory Data Analysis Statistics.

Deterministic multivariate statistical analysis for a Pandas DataFrame.

Features
--------
- Multivariate descriptive statistics
- Correlation matrix
- Covariance matrix
- Pairwise feature relationships
- Variance Inflation Factor (VIF)
- Principal Component Analysis (PCA)
- Feature interaction analysis
- Multivariate outlier detection
- Missingness pattern analysis
- Decision Tree feature importance
- Logistic Regression feature importance
- K-Means clustering
- Cluster profiling

All statistical calculations are performed deterministically using
Pandas, NumPy, SciPy, and scikit-learn.

LangChain tools are thin wrappers around the deterministic methods.
"""

from __future__ import annotations

from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from langchain_core.tools import tool
from src.Utils.exception_handler import CustomException
from src.Utils.logger_setup import get_log, track_performance


class MultivariateStatistics:
    """
    Deterministic multivariate statistical analysis engine.

    Parameters
    ----------
    df:
        Input Pandas DataFrame.

    Notes
    -----
    The DataFrame is retained internally and is never modified.
    """

    DEFAULT_CORRELATION_METHOD = "pearson"
    DEFAULT_TOP_N = 10
    DEFAULT_VIF_THRESHOLD = 5.0
    DEFAULT_PCA_COMPONENTS = 5
    DEFAULT_OUTLIER_THRESHOLD = 3.0
    DEFAULT_N_CLUSTERS = 3
    DEFAULT_RANDOM_STATE = 42
    DEFAULT_KMEANS_N_INIT = 10
    DEFAULT_LOGISTIC_C = 1.0
    DEFAULT_LOGISTIC_MAX_ITER = 1000

    def __init__(self, df: pd.DataFrame) -> None:
        """
        Initialize the multivariate statistics engine.

        Parameters
        ----------
        df:
            Input Pandas DataFrame.

        Raises
        ------
        CustomException
            If the DataFrame is invalid.
        """
        try:
            if not isinstance(df, pd.DataFrame):
                raise ValueError("Input must be a Pandas DataFrame.")
            if df.empty:
                raise ValueError("Input DataFrame cannot be empty.")
            self.df = df
            self.logger = get_log(__name__)
            self.logger.info(
                "Initialized MultivariateStatistics with shape=%s.", df.shape
            )
        except Exception as exc:
            raise CustomException(exc) from exc

    def _validate_numeric_columns(
        self, columns: list[str] | None = None
    ) -> list[str]:
        """
        Validate and return numeric columns.

        Parameters
        ----------
        columns:
            Optional list of requested columns.

        Returns
        -------
        list[str]
            Valid numeric columns.
        """
        try:
            if columns is None:
                numeric_columns = self.df.select_dtypes(
                    include=np.number
                ).columns.tolist()
            else:
                if not columns:
                    raise ValueError("At least one column must be provided.")
                missing = [c for c in columns if c not in self.df.columns]
                if missing:
                    raise ValueError(f"Columns not found: {missing}")
                non_numeric = [
                    c for c in columns
                    if not pd.api.types.is_numeric_dtype(self.df[c])
                ]
                if non_numeric:
                    raise ValueError(f"Columns must be numeric: {non_numeric}")
                numeric_columns = columns.copy()
            if not numeric_columns:
                raise ValueError(
                    "No numeric columns available for multivariate analysis."
                )
            return numeric_columns
        except Exception as exc:
            raise CustomException(exc) from exc

    def _validate_target_column(self, target_column: str) -> None:
        """
        Validate target column.

        Parameters
        ----------
        target_column:
            Target column name.
        """
        try:
            if target_column not in self.df.columns:
                raise ValueError(f"Target column not found: {target_column}")
        except Exception as exc:
            raise CustomException(exc) from exc

    def _prepare_numeric_data(
        self, columns: list[str] | None = None
    ) -> pd.DataFrame:
        """
        Prepare complete-case numeric data.

        Parameters
        ----------
        columns:
            Numeric columns to analyze.

        Returns
        -------
        pd.DataFrame
            Numeric DataFrame with rows containing missing values
            removed.
        """
        try:
            numeric_columns = self._validate_numeric_columns(columns)
            data = self.df[numeric_columns].dropna()
            if data.empty:
                raise ValueError(
                    "No complete observations remain after removing missing values."
                )
            if len(data) < 2:
                raise ValueError("At least two observations are required.")
            return data
        except Exception as exc:
            raise CustomException(exc) from exc

    @track_performance
    def multivariate_summary(
        self, columns: list[str] | None = None
    ) -> dict[str, Any]:
        """
        Generate descriptive statistics for multiple numeric features.

        Parameters
        ----------
        columns:
            Numeric columns to analyze.

        Returns
        -------
        dict[str, Any]
            Descriptive statistics.
        """
        try:
            data = self._prepare_numeric_data(columns)
            summary = pd.DataFrame({
                "count": data.count(),
                "mean": data.mean(),
                "std": data.std(),
                "variance": data.var(),
                "min": data.min(),
                "q25": data.quantile(0.25),
                "median": data.median(),
                "q75": data.quantile(0.75),
                "max": data.max(),
                "skewness": data.skew(),
                "kurtosis": data.kurt(),
            })
            return {
                "analysis": "multivariate_summary",
                "columns": data.columns.tolist(),
                "observations": int(len(data)),
                "statistics": summary.to_dict(),
            }
        except Exception as exc:
            raise CustomException(exc) from exc

    @track_performance
    def correlation_matrix(
        self,
        columns: list[str] | None = None,
        method: str = DEFAULT_CORRELATION_METHOD,
    ) -> dict[str, Any]:
        """
        Calculate a multivariate correlation matrix.

        Parameters
        ----------
        columns:
            Numeric columns.
        method:
            Pearson, Spearman, or Kendall.

        Returns
        -------
        dict[str, Any]
            Correlation matrix.
        """
        try:
            valid_methods = {"pearson", "spearman", "kendall"}
            if method not in valid_methods:
                raise ValueError(
                    f"Unsupported correlation method: {method}. "
                    f"Choose from {sorted(valid_methods)}."
                )
            data = self._prepare_numeric_data(columns)
            matrix = data.corr(method=method)
            return {
                "analysis": "correlation_matrix",
                "method": method,
                "columns": data.columns.tolist(),
                "observations": int(len(data)),
                "matrix": matrix.to_dict(),
            }
        except Exception as exc:
            raise CustomException(exc) from exc

    @track_performance
    def covariance_matrix(
        self, columns: list[str] | None = None
    ) -> dict[str, Any]:
        """
        Calculate covariance matrix across numeric features.

        Parameters
        ----------
        columns:
            Numeric columns.

        Returns
        -------
        dict[str, Any]
            Covariance matrix.
        """
        try:
            data = self._prepare_numeric_data(columns)
            matrix = data.cov()
            return {
                "analysis": "covariance_matrix",
                "columns": data.columns.tolist(),
                "observations": int(len(data)),
                "matrix": matrix.to_dict(),
            }
        except Exception as exc:
            raise CustomException(exc) from exc

    @track_performance
    def pairwise_relationships(
        self,
        columns: list[str] | None = None,
        method: str = DEFAULT_CORRELATION_METHOD,
        top_n: int = DEFAULT_TOP_N,
    ) -> dict[str, Any]:
        """
        Calculate and rank pairwise feature relationships.

        Parameters
        ----------
        columns:
            Numeric columns.
        method:
            Pearson or Spearman.
        top_n:
            Number of strongest relationships.

        Returns
        -------
        dict[str, Any]
            Ranked pairwise relationships.
        """
        try:
            if method not in {"pearson", "spearman"}:
                raise ValueError("method must be 'pearson' or 'spearman'.")
            if top_n <= 0:
                raise ValueError("top_n must be greater than zero.")
            data = self._prepare_numeric_data(columns)
            if len(data.columns) < 2:
                raise ValueError("At least two numeric columns are required.")
            relationships: list[dict[str, Any]] = []
            for column_1, column_2 in combinations(data.columns, 2):
                x, y = data[column_1], data[column_2]
                if x.nunique() < 2 or y.nunique() < 2:
                    continue
                if method == "pearson":
                    coefficient, p_value = pearsonr(x, y)
                else:
                    coefficient, p_value = spearmanr(x, y)
                relationships.append({
                    "feature_1": column_1,
                    "feature_2": column_2,
                    "correlation": float(coefficient),
                    "absolute_correlation": float(abs(coefficient)),
                    "p_value": float(p_value),
                    "significant_at_0_05": bool(p_value < 0.05),
                })
            relationships.sort(
                key=lambda item: item["absolute_correlation"], reverse=True
            )
            return {
                "analysis": "pairwise_relationships",
                "method": method,
                "columns": data.columns.tolist(),
                "observations": int(len(data)),
                "top_relationships": relationships[:top_n],
                "total_relationships": len(relationships),
            }
        except Exception as exc:
            raise CustomException(exc) from exc

    @track_performance
    def variance_inflation_factor(
        self,
        columns: list[str] | None = None,
        threshold: float = DEFAULT_VIF_THRESHOLD,
    ) -> dict[str, Any]:
        """
        Calculate Variance Inflation Factor for numeric features.

        Parameters
        ----------
        columns:
            Numeric columns.
        threshold:
            Threshold used to flag potential multicollinearity.

        Returns
        -------
        dict[str, Any]
            VIF results.
        """
        try:
            if threshold <= 0:
                raise ValueError("threshold must be greater than zero.")
            data = self._prepare_numeric_data(columns)
            if len(data.columns) < 2:
                raise ValueError(
                    "At least two numeric columns are required for VIF."
                )
            values = data.to_numpy(dtype=float)
            results: list[dict[str, Any]] = []
            for index, column in enumerate(data.columns):
                y = values[:, index]
                other_indices = [i for i in range(values.shape[1]) if i != index]
                x = values[:, other_indices]
                x_design = np.column_stack([np.ones(len(x)), x])
                coefficients, *_ = np.linalg.lstsq(x_design, y, rcond=None)
                predicted = x_design @ coefficients
                residual_sum_squares = np.sum((y - predicted) ** 2)
                total_sum_squares = np.sum((y - np.mean(y)) ** 2)
                if np.isclose(total_sum_squares, 0):
                    vif = np.inf
                else:
                    r_squared = 1 - residual_sum_squares / total_sum_squares
                    denominator = 1 - r_squared
                    vif = np.inf if np.isclose(denominator, 0) else 1 / denominator
                results.append({
                    "feature": column,
                    "vif": float(vif),
                    "high_multicollinearity": bool(vif >= threshold),
                })
            results.sort(
                key=lambda item: (np.isinf(item["vif"]), item["vif"]),
                reverse=True,
            )
            return {
                "analysis": "variance_inflation_factor",
                "columns": data.columns.tolist(),
                "observations": int(len(data)),
                "threshold": threshold,
                "results": results,
            }
        except Exception as exc:
            raise CustomException(exc) from exc

    @track_performance
    def principal_component_analysis(
        self,
        columns: list[str] | None = None,
        n_components: int | None = None,
    ) -> dict[str, Any]:
        """
        Perform standardized Principal Component Analysis.

        Parameters
        ----------
        columns:
            Numeric columns.
        n_components:
            Number of principal components.

        Returns
        -------
        dict[str, Any]
            PCA results including explained variance and loadings.
        """
        try:
            data = self._prepare_numeric_data(columns)
            if len(data.columns) < 2:
                raise ValueError(
                    "At least two numeric columns are required for PCA."
                )
            max_components = min(data.shape[0], data.shape[1])
            if n_components is None:
                n_components = min(self.DEFAULT_PCA_COMPONENTS, max_components)
            if n_components <= 0:
                raise ValueError("n_components must be greater than zero.")
            if n_components > max_components:
                raise ValueError(
                    f"n_components cannot exceed {max_components}."
                )
            scaler = StandardScaler()
            standardized = scaler.fit_transform(data)
            pca = PCA(n_components=n_components)
            transformed = pca.fit_transform(standardized)
            explained_variance_ratio = pca.explained_variance_ratio_
            cumulative_variance = np.cumsum(explained_variance_ratio)
            component_names = [f"PC{i + 1}" for i in range(n_components)]
            loadings = pd.DataFrame(
                pca.components_.T, index=data.columns, columns=component_names
            )
            component_summary = [
                {
                    "component": f"PC{i + 1}",
                    "explained_variance_ratio": float(explained_variance_ratio[i]),
                    "cumulative_explained_variance": float(cumulative_variance[i]),
                }
                for i in range(n_components)
            ]
            return {
                "analysis": "principal_component_analysis",
                "columns": data.columns.tolist(),
                "observations": int(len(data)),
                "n_components": n_components,
                "explained_variance_ratio": explained_variance_ratio.tolist(),
                "cumulative_explained_variance": cumulative_variance.tolist(),
                "component_summary": component_summary,
                "loadings": loadings.to_dict(),
                "transformed_shape": list(transformed.shape),
            }
        except Exception as exc:
            raise CustomException(exc) from exc

    @track_performance
    def feature_interactions(
        self,
        columns: list[str] | None = None,
        top_n: int = DEFAULT_TOP_N,
    ) -> dict[str, Any]:
        """
        Analyze pairwise multiplicative feature interactions.

        Parameters
        ----------
        columns:
            Numeric columns.
        top_n:
            Number of interactions to return.

        Returns
        -------
        dict[str, Any]
            Interaction statistics.
        """
        try:
            if top_n <= 0:
                raise ValueError("top_n must be greater than zero.")
            data = self._prepare_numeric_data(columns)
            if len(data.columns) < 2:
                raise ValueError("At least two numeric columns are required.")
            interactions: list[dict[str, Any]] = []
            for column_1, column_2 in combinations(data.columns, 2):
                product_values = data[column_1] * data[column_2]
                interactions.append({
                    "feature_1": column_1,
                    "feature_2": column_2,
                    "interaction_mean": float(product_values.mean()),
                    "interaction_std": float(product_values.std()),
                    "interaction_min": float(product_values.min()),
                    "interaction_median": float(product_values.median()),
                    "interaction_max": float(product_values.max()),
                })
            interactions.sort(
                key=lambda item: abs(item["interaction_mean"]), reverse=True
            )
            return {
                "analysis": "feature_interactions",
                "columns": data.columns.tolist(),
                "observations": int(len(data)),
                "top_interactions": interactions[:top_n],
                "total_interactions": len(interactions),
            }
        except Exception as exc:
            raise CustomException(exc) from exc

    @track_performance
    def multivariate_outliers(
        self,
        columns: list[str] | None = None,
        threshold: float = DEFAULT_OUTLIER_THRESHOLD,
    ) -> dict[str, Any]:
        """
        Detect multivariate outliers using standardized distance.

        Parameters
        ----------
        columns:
            Numeric columns.
        threshold:
            Standardized Euclidean distance threshold.

        Returns
        -------
        dict[str, Any]
            Outlier information.
        """
        try:
            if threshold <= 0:
                raise ValueError("threshold must be greater than zero.")
            data = self._prepare_numeric_data(columns)
            if len(data.columns) < 2:
                raise ValueError("At least two numeric columns are required.")
            scaler = StandardScaler()
            standardized = scaler.fit_transform(data)
            distances = np.linalg.norm(standardized, axis=1)
            outlier_mask = distances > threshold
            outlier_indices = data.index[outlier_mask].tolist()
            return {
                "analysis": "multivariate_outliers",
                "columns": data.columns.tolist(),
                "observations": int(len(data)),
                "threshold": threshold,
                "outlier_count": int(outlier_mask.sum()),
                "outlier_percentage": float(outlier_mask.mean() * 100),
                "normal_count": int((~outlier_mask).sum()),
                "outlier_indices": outlier_indices,
                "distance_statistics": {
                    "mean": float(np.mean(distances)),
                    "median": float(np.median(distances)),
                    "std": float(np.std(distances)),
                    "max": float(np.max(distances)),
                },
            }
        except Exception as exc:
            raise CustomException(exc) from exc

    @track_performance
    def missingness_patterns(
        self,
        columns: list[str] | None = None,
        top_n: int = DEFAULT_TOP_N,
    ) -> dict[str, Any]:
        """
        Analyze recurring missing-value combinations.

        Parameters
        ----------
        columns:
            Numeric columns.
        top_n:
            Number of common patterns to return.

        Returns
        -------
        dict[str, Any]
            Missingness patterns.
        """
        try:
            if top_n <= 0:
                raise ValueError("top_n must be greater than zero.")
            numeric_columns = self._validate_numeric_columns(columns)
            data = self.df[numeric_columns]
            missing_mask = data.isna()
            pattern_strings = missing_mask.apply(
                lambda row: "".join("1" if v else "0" for v in row), axis=1
            )
            pattern_counts = pattern_strings.value_counts().head(top_n)
            patterns = []
            for pattern, count in pattern_counts.items():
                missing_columns = [
                    col for col, flag in zip(numeric_columns, pattern)
                    if flag == "1"
                ]
                patterns.append({
                    "pattern": pattern,
                    "count": int(count),
                    "percentage": float(count / len(data) * 100),
                    "missing_columns": missing_columns,
                })
            return {
                "analysis": "missingness_patterns",
                "columns": numeric_columns,
                "observations": int(len(data)),
                "unique_patterns": int(pattern_strings.nunique()),
                "top_patterns": patterns,
            }
        except Exception as exc:
            raise CustomException(exc) from exc

    @track_performance
    def decision_tree_feature_importance(
        self,
        target_column: str,
        feature_columns: list[str] | None = None,
        max_depth: int | None = None,
        random_state: int = DEFAULT_RANDOM_STATE,
    ) -> dict[str, Any]:
        """
        Calculate Decision Tree feature importance.

        Parameters
        ----------
        target_column:
            Classification target.

        feature_columns:
            Numeric predictor columns.
            If None, all numeric columns except target are used.

        max_depth:
            Maximum tree depth.

        random_state:
            Random seed.

        Returns
        -------
        dict[str, Any]
            Feature importance results.
        """
        try:
            self._validate_target_column(target_column)
            if feature_columns is None:
                feature_columns = [
                    c for c in self.df.select_dtypes(include=np.number).columns
                    if c != target_column
                ]
            feature_columns = self._validate_numeric_columns(feature_columns)
            if target_column in feature_columns:
                raise ValueError(
                    "Target column cannot also be a feature column."
                )
            if max_depth is not None and max_depth <= 0:
                raise ValueError(
                    "max_depth must be greater than zero when provided."
                )
            data = self.df[feature_columns + [target_column]].dropna()
            if data.empty:
                raise ValueError("No complete observations available.")
            if data[target_column].nunique() < 2:
                raise ValueError("Target must contain at least two classes.")
            if len(data) < 10:
                raise ValueError(
                    "At least 10 observations are recommended "
                    "for Decision Tree analysis."
                )
            x, y = data[feature_columns], data[target_column]
            model = DecisionTreeClassifier(
                max_depth=max_depth, random_state=random_state
            )
            model.fit(x, y)
            importance = pd.Series(
                model.feature_importances_, index=feature_columns
            ).sort_values(ascending=False)
            results = [
                {"feature": feature, "importance": float(value), "rank": rank}
                for rank, (feature, value) in enumerate(
                    importance.items(), start=1
                )
            ]
            return {
                "analysis": "decision_tree_feature_importance",
                "target": target_column,
                "features": feature_columns,
                "observations": int(len(data)),
                "classes": y.unique().tolist(),
                "n_classes": int(y.nunique()),
                "max_depth": max_depth,
                "random_state": random_state,
                "results": results,
            }
        except Exception as exc:
            raise CustomException(exc) from exc

    @track_performance
    def logistic_regression_feature_importance(
        self,
        target_column: str,
        feature_columns: list[str] | None = None,
        C: float = DEFAULT_LOGISTIC_C,
        max_iter: int = DEFAULT_LOGISTIC_MAX_ITER,
        random_state: int = DEFAULT_RANDOM_STATE,
    ) -> dict[str, Any]:
        """
        Calculate standardized Logistic Regression feature importance.

        Parameters
        ----------
        target_column:
            Binary classification target.

        feature_columns:
            Numeric predictor columns.

        C:
            Inverse regularization strength.

        max_iter:
            Maximum optimization iterations.

        random_state:
            Random seed.

        Returns
        -------
        dict[str, Any]
            Standardized coefficient-based feature importance.
        """
        try:
            self._validate_target_column(target_column)
            if feature_columns is None:
                feature_columns = [
                    c for c in self.df.select_dtypes(include=np.number).columns
                    if c != target_column
                ]
            feature_columns = self._validate_numeric_columns(feature_columns)
            if target_column in feature_columns:
                raise ValueError(
                    "Target column cannot also be a feature column."
                )
            if C <= 0:
                raise ValueError("C must be greater than zero.")
            if max_iter <= 0:
                raise ValueError("max_iter must be greater than zero.")
            data = self.df[feature_columns + [target_column]].dropna()
            if data.empty:
                raise ValueError("No complete observations available.")
            if data[target_column].nunique() != 2:
                raise ValueError(
                    "Logistic Regression feature importance "
                    "requires exactly two target classes."
                )
            if len(data) < 10:
                raise ValueError(
                    "At least 10 observations are recommended "
                    "for Logistic Regression analysis."
                )
            x, y = data[feature_columns], data[target_column]
            scaler = StandardScaler()
            x_scaled = scaler.fit_transform(x)
            model = LogisticRegression(
                C=C, max_iter=max_iter, random_state=random_state
            )
            model.fit(x_scaled, y)
            coefficients = model.coef_[0]
            results = []
            for feature, coefficient in zip(feature_columns, coefficients):
                results.append({
                    "feature": feature,
                    "coefficient": float(coefficient),
                    "absolute_importance": float(abs(coefficient)),
                    "direction": (
                        "positive" if coefficient > 0
                        else "negative" if coefficient < 0
                        else "neutral"
                    ),
                })
            results.sort(key=lambda item: item["absolute_importance"], reverse=True)
            for rank, result in enumerate(results, start=1):
                result["rank"] = rank
            return {
                "analysis": "logistic_regression_feature_importance",
                "target": target_column,
                "features": feature_columns,
                "observations": int(len(data)),
                "classes": y.unique().tolist(),
                "C": C,
                "max_iter": max_iter,
                "random_state": random_state,
                "standardized": True,
                "results": results,
            }
        except Exception as exc:
            raise CustomException(exc) from exc

    @track_performance
    def kmeans_clustering(
        self,
        feature_columns: list[str] | None = None,
        n_clusters: int = DEFAULT_N_CLUSTERS,
        random_state: int = DEFAULT_RANDOM_STATE,
        n_init: int = DEFAULT_KMEANS_N_INIT,
    ) -> dict[str, Any]:
        """
        Perform K-Means clustering on numeric features.

        Parameters
        ----------
        feature_columns:
            Numeric features.
        n_clusters:
            Number of clusters.
        random_state:
            Random seed.
        n_init:
            Number of K-Means initializations.

        Returns
        -------
        dict[str, Any]
            Cluster assignments and quality metrics.
        """
        try:
            if n_clusters < 2:
                raise ValueError("n_clusters must be at least 2.")
            if n_init <= 0:
                raise ValueError("n_init must be greater than zero.")
            data = self._prepare_numeric_data(feature_columns)
            if len(data.columns) < 2:
                raise ValueError(
                    "At least two numeric features are required for clustering."
                )
            if len(data) <= n_clusters:
                raise ValueError(
                    "Number of observations must be greater than number of clusters."
                )
            scaler = StandardScaler()
            x_scaled = scaler.fit_transform(data)
            model = KMeans(
                n_clusters=n_clusters, random_state=random_state, n_init=n_init
            )
            labels = model.fit_predict(x_scaled)
            silhouette = silhouette_score(x_scaled, labels)
            cluster_counts = pd.Series(labels).value_counts().sort_index()
            centers_scaled = pd.DataFrame(
                model.cluster_centers_, columns=data.columns
            )
            centers_original = pd.DataFrame(
                scaler.inverse_transform(model.cluster_centers_),
                columns=data.columns,
            )
            return {
                "analysis": "kmeans_clustering",
                "features": data.columns.tolist(),
                "observations": int(len(data)),
                "n_clusters": n_clusters,
                "random_state": random_state,
                "n_init": n_init,
                "inertia": float(model.inertia_),
                "silhouette_score": float(silhouette),
                "cluster_counts": {
                    str(cluster): int(count)
                    for cluster, count in cluster_counts.items()
                },
                "cluster_centers_standardized": centers_scaled.to_dict(),
                "cluster_centers_original": centers_original.to_dict(),
                "cluster_assignments": [
                    {
                        "index": (
                            int(index)
                            if isinstance(index, (int, np.integer))
                            else str(index)
                        ),
                        "cluster": int(cluster),
                    }
                    for index, cluster in zip(data.index, labels)
                ],
            }
        except Exception as exc:
            raise CustomException(exc) from exc

    @track_performance
    def cluster_profile(
        self,
        feature_columns: list[str] | None = None,
        n_clusters: int = DEFAULT_N_CLUSTERS,
        random_state: int = DEFAULT_RANDOM_STATE,
        n_init: int = DEFAULT_KMEANS_N_INIT,
    ) -> dict[str, Any]:
        """
        Perform K-Means clustering and generate cluster profiles.

        Parameters
        ----------
        feature_columns:
            Numeric features.
        n_clusters:
            Number of clusters.
        random_state:
            Random seed.
        n_init:
            Number of K-Means initializations.

        Returns
        -------
        dict[str, Any]
            Cluster-level descriptive statistics.
        """
        try:
            if n_clusters < 2:
                raise ValueError("n_clusters must be at least 2.")
            if n_init <= 0:
                raise ValueError("n_init must be greater than zero.")
            data = self._prepare_numeric_data(feature_columns)
            if len(data.columns) < 2:
                raise ValueError(
                    "At least two numeric features are required for clustering."
                )
            if len(data) <= n_clusters:
                raise ValueError(
                    "Number of observations must be greater than number of clusters."
                )
            scaler = StandardScaler()
            x_scaled = scaler.fit_transform(data)
            model = KMeans(
                n_clusters=n_clusters, random_state=random_state, n_init=n_init
            )
            labels = model.fit_predict(x_scaled)
            clustered_data = data.copy()
            clustered_data["cluster"] = labels
            clusters: dict[str, Any] = {}
            for cluster in sorted(clustered_data["cluster"].unique()):
                cluster_data = clustered_data[clustered_data["cluster"] == cluster]
                feature_profile: dict[str, Any] = {}
                for feature in data.columns:
                    values = cluster_data[feature]
                    feature_profile[feature] = {
                        "count": int(values.count()),
                        "mean": float(values.mean()),
                        "median": float(values.median()),
                        "std": (
                            float(values.std()) if values.count() > 1 else None
                        ),
                        "min": float(values.min()),
                        "max": float(values.max()),
                    }
                clusters[str(cluster)] = {
                    "size": int(len(cluster_data)),
                    "percentage": float(
                        len(cluster_data) / len(clustered_data) * 100
                    ),
                    "features": feature_profile,
                }
            silhouette = silhouette_score(x_scaled, labels)
            return {
                "analysis": "cluster_profile",
                "features": data.columns.tolist(),
                "observations": int(len(data)),
                "n_clusters": n_clusters,
                "random_state": random_state,
                "n_init": n_init,
                "inertia": float(model.inertia_),
                "silhouette_score": float(silhouette),
                "clusters": clusters,
            }
        except Exception as exc:
            raise CustomException(exc) from exc

    def build_tools(self) -> list[Any]:
        """
        Build LangChain tools for multivariate statistics.

        Returns
        -------
        list[Any]
            LangChain-compatible statistical tools.
        """
        @tool
        def multivariate_summary(
            columns: list[str] | None = None,
        ) -> dict[str, Any]:
            """
            Generate descriptive statistics across multiple
            numeric features.
            """
            return self.multivariate_summary(columns=columns)

        @tool
        def correlation_matrix(
            columns: list[str] | None = None, method: str = "pearson"
        ) -> dict[str, Any]:
            """
            Calculate a correlation matrix across multiple
            numeric features.

            Supported methods:
            - pearson
            - spearman
            - kendall
            """
            return self.correlation_matrix(columns=columns, method=method)

        @tool
        def covariance_matrix(
            columns: list[str] | None = None,
        ) -> dict[str, Any]:
            """
            Calculate the covariance matrix across multiple
            numeric features.
            """
            return self.covariance_matrix(columns=columns)

        @tool
        def pairwise_relationships(
            columns: list[str] | None = None,
            method: str = "pearson",
            top_n: int = 10,
        ) -> dict[str, Any]:
            """
            Identify the strongest pairwise relationships
            among numeric features.
            """
            return self.pairwise_relationships(
                columns=columns, method=method, top_n=top_n
            )

        @tool
        def variance_inflation_factor(
            columns: list[str] | None = None, threshold: float = 5.0
        ) -> dict[str, Any]:
            """
            Calculate Variance Inflation Factor (VIF) to
            identify potential multicollinearity.
            """
            return self.variance_inflation_factor(
                columns=columns, threshold=threshold
            )

        @tool
        def principal_component_analysis(
            columns: list[str] | None = None, n_components: int | None = None
        ) -> dict[str, Any]:
            """
            Perform standardized PCA and return explained
            variance and feature loadings.
            """
            return self.principal_component_analysis(
                columns=columns, n_components=n_components
            )

        @tool
        def feature_interactions(
            columns: list[str] | None = None, top_n: int = 10
        ) -> dict[str, Any]:
            """
            Analyze pairwise multiplicative feature interactions.
            """
            return self.feature_interactions(columns=columns, top_n=top_n)

        @tool
        def multivariate_outliers(
            columns: list[str] | None = None, threshold: float = 3.0
        ) -> dict[str, Any]:
            """
            Detect multivariate outliers using standardized
            feature distance.
            """
            return self.multivariate_outliers(
                columns=columns, threshold=threshold
            )

        @tool
        def missingness_patterns(
            columns: list[str] | None = None, top_n: int = 10
        ) -> dict[str, Any]:
            """
            Analyze recurring missing-value combinations across
            multiple numeric features.
            """
            return self.missingness_patterns(columns=columns, top_n=top_n)

        @tool
        def decision_tree_feature_importance(
            target_column: str,
            feature_columns: list[str] | None = None,
            max_depth: int | None = None,
            random_state: int = 42,
        ) -> dict[str, Any]:
            """
            Calculate feature importance using a Decision Tree
            classifier.
            """
            return self.decision_tree_feature_importance(
                target_column=target_column,
                feature_columns=feature_columns,
                max_depth=max_depth,
                random_state=random_state,
            )

        @tool
        def logistic_regression_feature_importance(
            target_column: str,
            feature_columns: list[str] | None = None,
            C: float = 1.0,
            max_iter: int = 1000,
            random_state: int = 42,
        ) -> dict[str, Any]:
            """
            Calculate standardized Logistic Regression feature
            importance for a binary classification target.
            """
            return self.logistic_regression_feature_importance(
                target_column=target_column,
                feature_columns=feature_columns,
                C=C,
                max_iter=max_iter,
                random_state=random_state,
            )

        @tool
        def kmeans_clustering(
            feature_columns: list[str] | None = None,
            n_clusters: int = 3,
            random_state: int = 42,
            n_init: int = 10,
        ) -> dict[str, Any]:
            """
            Perform K-Means clustering on multiple numeric
            features.
            """
            return self.kmeans_clustering(
                feature_columns=feature_columns,
                n_clusters=n_clusters,
                random_state=random_state,
                n_init=n_init,
            )

        @tool
        def cluster_profile(
            feature_columns: list[str] | None = None,
            n_clusters: int = 3,
            random_state: int = 42,
            n_init: int = 10,
        ) -> dict[str, Any]:
            """
            Perform K-Means clustering and generate deterministic
            profiles for each cluster.
            """
            return self.cluster_profile(
                feature_columns=feature_columns,
                n_clusters=n_clusters,
                random_state=random_state,
                n_init=n_init,
            )

        tools: list[Any] = [
            multivariate_summary,
            correlation_matrix,
            covariance_matrix,
            pairwise_relationships,
            variance_inflation_factor,
            principal_component_analysis,
            feature_interactions,
            multivariate_outliers,
            missingness_patterns,
            decision_tree_feature_importance,
            logistic_regression_feature_importance,
            kmeans_clustering,
            cluster_profile,
        ]
        self.logger.info(
            "Built %d multivariate statistical tools.", len(tools)
        )
        return tools

#========================================================================

#====================================================
# Multivariate Visualization
#====================================================
"""
Multivariate Exploratory Data Analysis Visualizations.

Deterministic multivariate visualization tools for a Pandas DataFrame.

Visualizations
--------------
- Correlation heatmap
- Clustered correlation heatmap
- Pair plot
- PCA explained variance plot
- PCA loading plot
- Decision Tree feature importance
- Logistic Regression feature importance
- K-Means cluster scatter plot
- Cluster size plot
- Cluster profile heatmap
- Multivariate outlier distance plot
- Missingness pattern heatmap

All calculations required for visualization are performed
deterministically using Pandas, NumPy, Seaborn, Matplotlib,
and scikit-learn.

The DataFrame is retained internally and is never modified.
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from langchain_core.tools import tool
from src.Utils.exception_handler import CustomException
from src.Utils.logger_setup import get_log, track_performance

class MultivariateVisualization:
    """
    Deterministic multivariate visualization engine.

    Parameters
    ----------
    df:
        Input Pandas DataFrame.

    Notes
    -----
    The DataFrame is retained internally and is never modified.
    """

    DEFAULT_FIGSIZE = (12, 6)
    DEFAULT_TOP_N = 10
    DEFAULT_N_CLUSTERS = 3
    DEFAULT_RANDOM_STATE = 42
    DEFAULT_KMEANS_N_INIT = 10
    DEFAULT_PCA_COMPONENTS = 5
    DEFAULT_OUTLIER_THRESHOLD = 3.0

    def __init__(self, df: pd.DataFrame) -> None:
        """
        Initialize the multivariate visualization engine.

        Parameters
        ----------
        df:
            Input DataFrame.

        Raises
        ------
        CustomException
            If the DataFrame is invalid.
        """
        try:
            if not isinstance(df, pd.DataFrame):
                raise ValueError("Input must be a Pandas DataFrame.")
            if df.empty:
                raise ValueError("Input DataFrame cannot be empty.")
            self.df = df
            self.logger = get_log(__name__)
            self.logger.info(
                "Initialized MultivariateVisualization with shape=%s.",
                df.shape,
            )
        except Exception as exc:
            raise CustomException(exc) from exc

    def _validate_numeric_columns(
        self, columns: list[str] | None = None
    ) -> list[str]:
        """
        Validate numeric columns.

        Parameters
        ----------
        columns:
            Optional numeric columns.

        Returns
        -------
        list[str]
            Valid numeric columns.
        """
        try:
            if columns is None:
                numeric_columns = self.df.select_dtypes(
                    include=np.number
                ).columns.tolist()
            else:
                if not columns:
                    raise ValueError("At least one column must be provided.")
                missing = [c for c in columns if c not in self.df.columns]
                if missing:
                    raise ValueError(f"Columns not found: {missing}")
                non_numeric = [
                    c for c in columns
                    if not pd.api.types.is_numeric_dtype(self.df[c])
                ]
                if non_numeric:
                    raise ValueError(f"Columns must be numeric: {non_numeric}")
                numeric_columns = columns.copy()
            if not numeric_columns:
                raise ValueError("No numeric columns available.")
            return numeric_columns
        except Exception as exc:
            raise CustomException(exc) from exc

    def _validate_target_column(self, target_column: str) -> None:
        """
        Validate target column.

        Parameters
        ----------
        target_column:
            Target column name.
        """
        try:
            if target_column not in self.df.columns:
                raise ValueError(f"Target column not found: {target_column}")
        except Exception as exc:
            raise CustomException(exc) from exc

    def _prepare_numeric_data(
        self, columns: list[str] | None = None
    ) -> pd.DataFrame:
        """
        Prepare complete-case numeric data.

        Parameters
        ----------
        columns:
            Numeric columns.

        Returns
        -------
        pd.DataFrame
            Complete numeric observations.
        """
        try:
            numeric_columns = self._validate_numeric_columns(columns)
            data = self.df[numeric_columns].dropna()
            if data.empty:
                raise ValueError(
                    "No complete observations remain after removing missing values."
                )
            if len(data) < 2:
                raise ValueError("At least two observations are required.")
            return data
        except Exception as exc:
            raise CustomException(exc) from exc

    def _create_figure(
        self, figsize: tuple[int, int] | None = None
    ) -> plt.Figure:
        """
        Create a Matplotlib figure.

        Parameters
        ----------
        figsize:
            Figure size.

        Returns
        -------
        matplotlib.figure.Figure
            Created figure.
        """
        return plt.figure(figsize=figsize or self.DEFAULT_FIGSIZE)

    def _format_axis(
        self,
        ax: plt.Axes,
        title: str,
        xlabel: str | None = None,
        ylabel: str | None = None,
    ) -> None:
        """
        Format a Matplotlib axis.
        """
        ax.set_title(title, fontsize=14, fontweight="bold")
        if xlabel:
            ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25, linestyle="--")
        plt.tight_layout()

    @track_performance
    def correlation_heatmap(
        self,
        columns: list[str] | None = None,
        method: str = "pearson",
        figsize: tuple[int, int] | None = None,
        annot: bool = True,
    ) -> plt.Figure:
        """
        Plot a correlation heatmap.

        Parameters
        ----------
        columns:
            Numeric columns.
        method:
            Pearson, Spearman, or Kendall.
        figsize:
            Figure size.
        annot:
            Display correlation values.

        Returns
        -------
        matplotlib.figure.Figure
            Correlation heatmap.
        """
        try:
            valid_methods = {"pearson", "spearman", "kendall"}
            if method not in valid_methods:
                raise ValueError(f"Unsupported correlation method: {method}")
            data = self._prepare_numeric_data(columns)
            if len(data.columns) < 2:
                raise ValueError("At least two numeric columns are required.")
            correlation = data.corr(method=method)
            fig, ax = plt.subplots(figsize=figsize or self.DEFAULT_FIGSIZE)
            sns.heatmap(
                correlation,
                annot=annot,
                fmt=".2f",
                cmap="coolwarm",
                center=0,
                square=True,
                linewidths=0.5,
                ax=ax,
            )
            self._format_axis(ax, f"{method.title()} Correlation Heatmap")
            return fig
        except Exception as exc:
            raise CustomException(exc) from exc

    @track_performance
    def clustered_correlation_heatmap(
        self,
        columns: list[str] | None = None,
        method: str = "pearson",
        figsize: tuple[int, int] | None = None,
    ) -> plt.Figure:
        """
        Plot a hierarchical clustered correlation heatmap.

        Parameters
        ----------
        columns:
            Numeric columns.
        method:
            Pearson, Spearman, or Kendall.
        figsize:
            Figure size.

        Returns
        -------
        matplotlib.figure.Figure
            Seaborn ClusterGrid figure.
        """
        try:
            valid_methods = {"pearson", "spearman", "kendall"}
            if method not in valid_methods:
                raise ValueError(f"Unsupported correlation method: {method}")
            data = self._prepare_numeric_data(columns)
            if len(data.columns) < 2:
                raise ValueError("At least two numeric columns are required.")
            correlation = data.corr(method=method)
            grid = sns.clustermap(
                correlation,
                cmap="coolwarm",
                center=0,
                annot=True,
                fmt=".2f",
                figsize=figsize or self.DEFAULT_FIGSIZE,
                linewidths=0.5,
            )
            grid.figure.suptitle(
                f"Clustered {method.title()} Correlation Heatmap",
                y=1.02,
                fontsize=14,
                fontweight="bold",
            )
            return grid.figure
        except Exception as exc:
            raise CustomException(exc) from exc

    @track_performance
    def pair_plot(
        self,
        columns: list[str] | None = None,
        hue_column: str | None = None,
        max_columns: int = 5,
        diag_kind: str = "hist",
        corner: bool = False,
    ) -> sns.axisgrid.PairGrid:
        """
        Plot pairwise relationships among numeric features.

        Parameters
        ----------
        columns:
            Numeric columns.
        hue_column:
            Optional categorical/class column.
        max_columns:
            Maximum number of numeric columns.
        diag_kind:
            Histogram or KDE.
        corner:
            Plot only lower triangle.

        Returns
        -------
        seaborn.axisgrid.PairGrid
            Pair plot.
        """
        try:
            if max_columns < 2:
                raise ValueError("max_columns must be at least 2.")
            if diag_kind not in {"hist", "kde"}:
                raise ValueError("diag_kind must be 'hist' or 'kde'.")
            numeric_columns = self._validate_numeric_columns(columns)
            if len(numeric_columns) > max_columns:
                numeric_columns = numeric_columns[:max_columns]
            if len(numeric_columns) < 2:
                raise ValueError(
                    "At least two numeric columns are required for a pair plot."
                )
            selected_columns = numeric_columns.copy()
            if hue_column is not None:
                self._validate_target_column(hue_column)
                if hue_column not in selected_columns:
                    selected_columns.append(hue_column)
            data = self.df[selected_columns].dropna()
            grid = sns.pairplot(
                data,
                vars=numeric_columns,
                hue=hue_column,
                diag_kind=diag_kind,
                corner=corner,
            )
            grid.figure.suptitle(
                "Multivariate Pair Plot",
                y=1.02,
                fontsize=14,
                fontweight="bold",
            )
            return grid
        except Exception as exc:
            raise CustomException(exc) from exc

    @track_performance
    def pca_explained_variance_plot(
        self,
        columns: list[str] | None = None,
        n_components: int | None = None,
        figsize: tuple[int, int] | None = None,
    ) -> plt.Figure:
        """
        Plot PCA explained and cumulative explained variance.

        Parameters
        ----------
        columns:
            Numeric columns.
        n_components:
            Number of components.
        figsize:
            Figure size.

        Returns
        -------
        matplotlib.figure.Figure
            PCA explained variance plot.
        """
        try:
            data = self._prepare_numeric_data(columns)
            if len(data.columns) < 2:
                raise ValueError("At least two numeric columns are required.")
            max_components = min(data.shape[0], data.shape[1])
            if n_components is None:
                n_components = min(self.DEFAULT_PCA_COMPONENTS, max_components)
            if n_components <= 0:
                raise ValueError("n_components must be greater than zero.")
            if n_components > max_components:
                raise ValueError(
                    f"n_components cannot exceed {max_components}."
                )
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(data)
            pca = PCA(n_components=n_components)
            pca.fit(scaled_data)
            variance = pca.explained_variance_ratio_ * 100
            cumulative = np.cumsum(variance)
            components = [f"PC{i + 1}" for i in range(n_components)]
            fig, ax = plt.subplots(figsize=figsize or self.DEFAULT_FIGSIZE)
            ax.bar(components, variance, alpha=0.7, label="Explained variance")
            ax.plot(
                components,
                cumulative,
                marker="o",
                linewidth=2,
                label="Cumulative variance",
            )
            ax.set_ylim(0, min(100, max(cumulative.max() * 1.1, 10)))
            self._format_axis(
                ax,
                "PCA Explained Variance",
                "Principal component",
                "Variance explained (%)",
            )
            ax.legend()
            return fig
        except Exception as exc:
            raise CustomException(exc) from exc

    @track_performance
    def pca_loading_plot(
        self,
        columns: list[str] | None = None,
        component_1: int = 1,
        component_2: int = 2,
        figsize: tuple[int, int] | None = None,
    ) -> plt.Figure:
        """
        Plot feature loadings for two principal components.

        Parameters
        ----------
        columns:
            Numeric columns.
        component_1:
            First component number.
        component_2:
            Second component number.
        figsize:
            Figure size.

        Returns
        -------
        matplotlib.figure.Figure
            PCA loading plot.
        """
        try:
            if component_1 <= 0 or component_2 <= 0:
                raise ValueError("Component numbers must be positive.")
            if component_1 == component_2:
                raise ValueError(
                    "component_1 and component_2 must be different."
                )
            data = self._prepare_numeric_data(columns)
            if len(data.columns) < 2:
                raise ValueError("At least two numeric columns are required.")
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(data)
            pca = PCA(n_components=min(data.shape[0], data.shape[1]))
            pca.fit(scaled_data)
            max_component = pca.components_.shape[0]
            if component_1 > max_component:
                raise ValueError(
                    f"component_1 cannot exceed {max_component}."
                )
            if component_2 > max_component:
                raise ValueError(
                    f"component_2 cannot exceed {max_component}."
                )
            x = pca.components_[component_1 - 1]
            y = pca.components_[component_2 - 1]
            fig, ax = plt.subplots(figsize=figsize or self.DEFAULT_FIGSIZE)
            ax.scatter(x, y, s=80)
            for index, feature in enumerate(data.columns):
                ax.annotate(
                    feature,
                    (x[index], y[index]),
                    xytext=(5, 5),
                    textcoords="offset points",
                )
            ax.axhline(0, linewidth=0.8)
            ax.axvline(0, linewidth=0.8)
            self._format_axis(
                ax,
                "PCA Feature Loading Plot",
                f"PC{component_1} loading",
                f"PC{component_2} loading",
            )
            return fig
        except Exception as exc:
            raise CustomException(exc) from exc

    @track_performance
    def decision_tree_feature_importance_plot(
        self,
        target_column: str,
        feature_columns: list[str] | None = None,
        max_depth: int | None = None,
        random_state: int = DEFAULT_RANDOM_STATE,
        top_n: int = DEFAULT_TOP_N,
        figsize: tuple[int, int] | None = None,
    ) -> plt.Figure:
        """
        Plot Decision Tree feature importance.

        Parameters
        ----------
        target_column:
            Classification target.
        feature_columns:
            Numeric predictors.
        max_depth:
            Maximum tree depth.
        random_state:
            Random seed.
        top_n:
            Number of features to display.
        figsize:
            Figure size.

        Returns
        -------
        matplotlib.figure.Figure
            Feature importance plot.
        """
        try:
            self._validate_target_column(target_column)
            if top_n <= 0:
                raise ValueError("top_n must be greater than zero.")
            if feature_columns is None:
                feature_columns = [
                    c for c in self.df.select_dtypes(include=np.number).columns
                    if c != target_column
                ]
            feature_columns = self._validate_numeric_columns(feature_columns)
            data = self.df[feature_columns + [target_column]].dropna()
            if data.empty:
                raise ValueError("No complete observations available.")
            if data[target_column].nunique() < 2:
                raise ValueError("Target must contain at least two classes.")
            model = DecisionTreeClassifier(
                max_depth=max_depth, random_state=random_state
            )
            model.fit(data[feature_columns], data[target_column])
            importance = (
                pd.Series(model.feature_importances_, index=feature_columns)
                .sort_values(ascending=False)
                .head(top_n)
                .sort_values()
            )
            fig, ax = plt.subplots(figsize=figsize or self.DEFAULT_FIGSIZE)
            sns.barplot(x=importance.values, y=importance.index, ax=ax)
            self._format_axis(
                ax,
                "Decision Tree Feature Importance",
                "Feature importance",
                "Feature",
            )
            return fig
        except Exception as exc:
            raise CustomException(exc) from exc

    @track_performance
    def logistic_regression_feature_importance_plot(
        self,
        target_column: str,
        feature_columns: list[str] | None = None,
        C: float = 1.0,
        max_iter: int = 1000,
        random_state: int = DEFAULT_RANDOM_STATE,
        top_n: int = DEFAULT_TOP_N,
        figsize: tuple[int, int] | None = None,
    ) -> plt.Figure:
        """
        Plot standardized Logistic Regression coefficients.

        Parameters
        ----------
        target_column:
            Binary classification target.
        feature_columns:
            Numeric predictors.
        C:
            Inverse regularization strength.
        max_iter:
            Maximum iterations.
        random_state:
            Random seed.
        top_n:
            Number of features to display.
        figsize:
            Figure size.

        Returns
        -------
        matplotlib.figure.Figure
            Logistic coefficient plot.
        """
        try:
            self._validate_target_column(target_column)
            if top_n <= 0:
                raise ValueError("top_n must be greater than zero.")
            if C <= 0:
                raise ValueError("C must be greater than zero.")
            if max_iter <= 0:
                raise ValueError("max_iter must be greater than zero.")
            if feature_columns is None:
                feature_columns = [
                    c for c in self.df.select_dtypes(include=np.number).columns
                    if c != target_column
                ]
            feature_columns = self._validate_numeric_columns(feature_columns)
            data = self.df[feature_columns + [target_column]].dropna()
            if data.empty:
                raise ValueError("No complete observations available.")
            if data[target_column].nunique() != 2:
                raise ValueError(
                    "Logistic Regression visualization "
                    "requires exactly two target classes."
                )
            scaler = StandardScaler()
            x_scaled = scaler.fit_transform(data[feature_columns])
            model = LogisticRegression(
                C=C, max_iter=max_iter, random_state=random_state
            )
            model.fit(x_scaled, data[target_column])
            coefficients = (
                pd.Series(model.coef_[0], index=feature_columns)
                .sort_values(key=np.abs, ascending=False)
                .head(top_n)
                .sort_values()
            )
            fig, ax = plt.subplots(figsize=figsize or self.DEFAULT_FIGSIZE)
            sns.barplot(x=coefficients.values, y=coefficients.index, ax=ax)
            ax.axvline(0, linewidth=1)
            self._format_axis(
                ax,
                "Logistic Regression Feature Coefficients",
                "Standardized coefficient",
                "Feature",
            )
            return fig
        except Exception as exc:
            raise CustomException(exc) from exc

    @track_performance
    def kmeans_cluster_scatter_plot(
        self,
        feature_columns: list[str] | None = None,
        x_column: str | None = None,
        y_column: str | None = None,
        n_clusters: int = DEFAULT_N_CLUSTERS,
        random_state: int = DEFAULT_RANDOM_STATE,
        n_init: int = DEFAULT_KMEANS_N_INIT,
        figsize: tuple[int, int] | None = None,
    ) -> plt.Figure:
        """
        Plot K-Means clusters using two numeric features.

        Parameters
        ----------
        feature_columns:
            Numeric features used for clustering.
        x_column:
            Feature for x-axis.
        y_column:
            Feature for y-axis.
        n_clusters:
            Number of clusters.
        random_state:
            Random seed.
        n_init:
            K-Means initializations.
        figsize:
            Figure size.

        Returns
        -------
        matplotlib.figure.Figure
            Cluster scatter plot.
        """
        try:
            if n_clusters < 2:
                raise ValueError("n_clusters must be at least 2.")
            data = self._prepare_numeric_data(feature_columns)
            if len(data.columns) < 2:
                raise ValueError("At least two numeric features are required.")
            if x_column is None:
                x_column = data.columns[0]
            if y_column is None:
                y_column = data.columns[1]
            if x_column not in data.columns:
                raise ValueError(f"x_column not found: {x_column}")
            if y_column not in data.columns:
                raise ValueError(f"y_column not found: {y_column}")
            if x_column == y_column:
                raise ValueError("x_column and y_column must be different.")
            scaler = StandardScaler()
            scaled = scaler.fit_transform(data)
            model = KMeans(
                n_clusters=n_clusters, random_state=random_state, n_init=n_init
            )
            labels = model.fit_predict(scaled)
            plot_data = data.copy()
            plot_data["cluster"] = labels.astype(str)
            fig, ax = plt.subplots(figsize=figsize or self.DEFAULT_FIGSIZE)
            sns.scatterplot(
                data=plot_data,
                x=x_column,
                y=y_column,
                hue="cluster",
                palette="tab10",
                s=70,
                alpha=0.75,
                ax=ax,
            )
            self._format_axis(
                ax, "K-Means Cluster Visualization", x_column, y_column
            )
            ax.legend(
                title="Cluster", bbox_to_anchor=(1.02, 1), loc="upper left"
            )
            return fig
        except Exception as exc:
            raise CustomException(exc) from exc

    @track_performance
    def cluster_size_plot(
        self,
        feature_columns: list[str] | None = None,
        n_clusters: int = DEFAULT_N_CLUSTERS,
        random_state: int = DEFAULT_RANDOM_STATE,
        n_init: int = DEFAULT_KMEANS_N_INIT,
        figsize: tuple[int, int] | None = None,
    ) -> plt.Figure:
        """
        Plot the number of observations in each K-Means cluster.

        Parameters
        ----------
        feature_columns:
            Numeric features.
        n_clusters:
            Number of clusters.
        random_state:
            Random seed.
        n_init:
            Number of K-Means initializations.
        figsize:
            Figure size.

        Returns
        -------
        matplotlib.figure.Figure
            Cluster size plot.
        """
        try:
            if n_clusters < 2:
                raise ValueError("n_clusters must be at least 2.")
            data = self._prepare_numeric_data(feature_columns)
            if len(data) <= n_clusters:
                raise ValueError(
                    "Number of observations must be greater than number of clusters."
                )
            scaler = StandardScaler()
            scaled = scaler.fit_transform(data)
            model = KMeans(
                n_clusters=n_clusters, random_state=random_state, n_init=n_init
            )
            labels = model.fit_predict(scaled)
            counts = pd.Series(labels).value_counts().sort_index()
            plot_data = pd.DataFrame({
                "cluster": counts.index.astype(str),
                "count": counts.values,
            })
            fig, ax = plt.subplots(figsize=figsize or self.DEFAULT_FIGSIZE)
            sns.barplot(data=plot_data, x="cluster", y="count", ax=ax)
            self._format_axis(
                ax,
                "K-Means Cluster Sizes",
                "Cluster",
                "Number of observations",
            )
            return fig
        except Exception as exc:
            raise CustomException(exc) from exc

    @track_performance
    def cluster_profile_heatmap(
        self,
        feature_columns: list[str] | None = None,
        n_clusters: int = DEFAULT_N_CLUSTERS,
        random_state: int = DEFAULT_RANDOM_STATE,
        n_init: int = DEFAULT_KMEANS_N_INIT,
        figsize: tuple[int, int] | None = None,
    ) -> plt.Figure:
        """
        Plot standardized cluster feature profiles.

        Parameters
        ----------
        feature_columns:
            Numeric features.
        n_clusters:
            Number of clusters.
        random_state:
            Random seed.
        n_init:
            Number of K-Means initializations.
        figsize:
            Figure size.

        Returns
        -------
        matplotlib.figure.Figure
            Cluster profile heatmap.
        """
        try:
            if n_clusters < 2:
                raise ValueError("n_clusters must be at least 2.")
            data = self._prepare_numeric_data(feature_columns)
            if len(data.columns) < 2:
                raise ValueError("At least two numeric features are required.")
            scaler = StandardScaler()
            scaled = scaler.fit_transform(data)
            scaled_df = pd.DataFrame(
                scaled, columns=data.columns, index=data.index
            )
            model = KMeans(
                n_clusters=n_clusters, random_state=random_state, n_init=n_init
            )
            labels = model.fit_predict(scaled)
            scaled_df["cluster"] = labels
            profile = scaled_df.groupby("cluster", observed=True).mean()
            fig, ax = plt.subplots(figsize=figsize or self.DEFAULT_FIGSIZE)
            sns.heatmap(
                profile,
                annot=True,
                fmt=".2f",
                cmap="coolwarm",
                center=0,
                linewidths=0.5,
                ax=ax,
            )
            self._format_axis(
                ax,
                "Standardized Cluster Feature Profiles",
                "Feature",
                "Cluster",
            )
            return fig
        except Exception as exc:
            raise CustomException(exc) from exc

    @track_performance
    def multivariate_outlier_distance_plot(
        self,
        columns: list[str] | None = None,
        threshold: float = DEFAULT_OUTLIER_THRESHOLD,
        figsize: tuple[int, int] | None = None,
    ) -> plt.Figure:
        """
        Plot standardized multivariate distances.

        Parameters
        ----------
        columns:
            Numeric columns.
        threshold:
            Distance threshold.
        figsize:
            Figure size.

        Returns
        -------
        matplotlib.figure.Figure
            Outlier distance plot.
        """
        try:
            if threshold <= 0:
                raise ValueError("threshold must be greater than zero.")
            data = self._prepare_numeric_data(columns)
            if len(data.columns) < 2:
                raise ValueError("At least two numeric columns are required.")
            scaler = StandardScaler()
            scaled = scaler.fit_transform(data)
            distances = np.linalg.norm(scaled, axis=1)
            outlier_mask = distances > threshold
            fig, ax = plt.subplots(figsize=figsize or self.DEFAULT_FIGSIZE)
            observation_numbers = np.arange(len(data))
            sns.scatterplot(
                x=observation_numbers[~outlier_mask],
                y=distances[~outlier_mask],
                ax=ax,
                label="Normal",
            )
            if outlier_mask.any():
                sns.scatterplot(
                    x=observation_numbers[outlier_mask],
                    y=distances[outlier_mask],
                    ax=ax,
                    label="Potential outlier",
                )
            ax.axhline(
                threshold,
                linestyle="--",
                linewidth=1.5,
                label=f"Threshold = {threshold}",
            )
            self._format_axis(
                ax,
                "Multivariate Outlier Distance",
                "Observation",
                "Standardized distance",
            )
            ax.legend()
            return fig
        except Exception as exc:
            raise CustomException(exc) from exc

    @track_performance
    def missingness_pattern_heatmap(
        self,
        columns: list[str] | None = None,
        figsize: tuple[int, int] | None = None,
        max_rows: int = 500,
    ) -> plt.Figure:
        """
        Plot missingness patterns across numeric features.

        Parameters
        ----------
        columns:
            Numeric columns.
        figsize:
            Figure size.
        max_rows:
            Maximum rows displayed.

        Returns
        -------
        matplotlib.figure.Figure
            Missingness heatmap.
        """
        try:
            if max_rows <= 0:
                raise ValueError("max_rows must be greater than zero.")
            numeric_columns = self._validate_numeric_columns(columns)
            data = self.df[numeric_columns].isna()
            if len(data) > max_rows:
                data = data.head(max_rows)
            fig, ax = plt.subplots(figsize=figsize or self.DEFAULT_FIGSIZE)
            sns.heatmap(
                data,
                cmap=["white", "black"],
                cbar=False,
                yticklabels=False,
                ax=ax,
            )
            self._format_axis(
                ax,
                "Multivariate Missingness Pattern",
                "Feature",
                "Observation",
            )
            return fig
        except Exception as exc:
            raise CustomException(exc) from exc

    def build_tools(self) -> list[Any]:
        """
        Build LangChain visualization tools.

        Returns
        -------
        list[Any]
            Multivariate visualization tools.
        """
        @tool
        def correlation_heatmap(
            columns: list[str] | None = None,
            method: str = "pearson",
            annot: bool = True,
        ) -> plt.Figure:
            """
            Create a correlation heatmap for multiple numeric
            features.
            """
            return self.correlation_heatmap(
                columns=columns, method=method, annot=annot
            )

        @tool
        def clustered_correlation_heatmap(
            columns: list[str] | None = None, method: str = "pearson"
        ) -> plt.Figure:
            """
            Create a hierarchical clustered correlation heatmap.
            """
            return self.clustered_correlation_heatmap(
                columns=columns, method=method
            )

        @tool
        def pair_plot(
            columns: list[str] | None = None,
            hue_column: str | None = None,
            max_columns: int = 5,
            diag_kind: str = "hist",
            corner: bool = False,
        ) -> Any:
            """
            Create a multivariate pair plot showing pairwise
            feature relationships.
            """
            return self.pair_plot(
                columns=columns,
                hue_column=hue_column,
                max_columns=max_columns,
                diag_kind=diag_kind,
                corner=corner,
            )

        @tool
        def pca_explained_variance_plot(
            columns: list[str] | None = None, n_components: int | None = None
        ) -> plt.Figure:
            """
            Plot PCA explained and cumulative explained variance.
            """
            return self.pca_explained_variance_plot(
                columns=columns, n_components=n_components
            )

        @tool
        def pca_loading_plot(
            columns: list[str] | None = None,
            component_1: int = 1,
            component_2: int = 2,
        ) -> plt.Figure:
            """
            Plot feature loadings for two principal components.
            """
            return self.pca_loading_plot(
                columns=columns,
                component_1=component_1,
                component_2=component_2,
            )

        @tool
        def decision_tree_feature_importance_plot(
            target_column: str,
            feature_columns: list[str] | None = None,
            max_depth: int | None = None,
            random_state: int = 42,
            top_n: int = 10,
        ) -> plt.Figure:
            """
            Plot Decision Tree feature importance.
            """
            return self.decision_tree_feature_importance_plot(
                target_column=target_column,
                feature_columns=feature_columns,
                max_depth=max_depth,
                random_state=random_state,
                top_n=top_n,
            )

        @tool
        def logistic_regression_feature_importance_plot(
            target_column: str,
            feature_columns: list[str] | None = None,
            C: float = 1.0,
            max_iter: int = 1000,
            random_state: int = 42,
            top_n: int = 10,
        ) -> plt.Figure:
            """
            Plot standardized Logistic Regression coefficients.
            """
            return self.logistic_regression_feature_importance_plot(
                target_column=target_column,
                feature_columns=feature_columns,
                C=C,
                max_iter=max_iter,
                random_state=random_state,
                top_n=top_n,
            )

        @tool
        def kmeans_cluster_scatter_plot(
            feature_columns: list[str] | None = None,
            x_column: str | None = None,
            y_column: str | None = None,
            n_clusters: int = 3,
            random_state: int = 42,
            n_init: int = 10,
        ) -> plt.Figure:
            """
            Plot K-Means clusters using two numeric features.
            """
            return self.kmeans_cluster_scatter_plot(
                feature_columns=feature_columns,
                x_column=x_column,
                y_column=y_column,
                n_clusters=n_clusters,
                random_state=random_state,
                n_init=n_init,
            )

        @tool
        def cluster_size_plot(
            feature_columns: list[str] | None = None,
            n_clusters: int = 3,
            random_state: int = 42,
            n_init: int = 10,
        ) -> plt.Figure:
            """
            Plot the number of observations in each K-Means
            cluster.
            """
            return self.cluster_size_plot(
                feature_columns=feature_columns,
                n_clusters=n_clusters,
                random_state=random_state,
                n_init=n_init,
            )

        @tool
        def cluster_profile_heatmap(
            feature_columns: list[str] | None = None,
            n_clusters: int = 3,
            random_state: int = 42,
            n_init: int = 10,
        ) -> plt.Figure:
            """
            Plot standardized feature profiles for K-Means
            clusters.
            """
            return self.cluster_profile_heatmap(
                feature_columns=feature_columns,
                n_clusters=n_clusters,
                random_state=random_state,
                n_init=n_init,
            )

        @tool
        def multivariate_outlier_distance_plot(
            columns: list[str] | None = None, threshold: float = 3.0
        ) -> plt.Figure:
            """
            Plot standardized multivariate observation distances
            and highlight potential outliers.
            """
            return self.multivariate_outlier_distance_plot(
                columns=columns, threshold=threshold
            )

        @tool
        def missingness_pattern_heatmap(
            columns: list[str] | None = None, max_rows: int = 500
        ) -> plt.Figure:
            """
            Visualize missing-value patterns across multiple
            numeric features.
            """
            return self.missingness_pattern_heatmap(
                columns=columns, max_rows=max_rows
            )

        tools: list[Any] = [
            correlation_heatmap,
            clustered_correlation_heatmap,
            pair_plot,
            pca_explained_variance_plot,
            pca_loading_plot,
            decision_tree_feature_importance_plot,
            logistic_regression_feature_importance_plot,
            kmeans_cluster_scatter_plot,
            cluster_size_plot,
            cluster_profile_heatmap,
            multivariate_outlier_distance_plot,
            missingness_pattern_heatmap,
        ]
        self.logger.info(
            "Built %d multivariate visualization tools.", len(tools)
        )
        return tools

#===============================================================================

# ====================================================
# Factory
# ====================================================

def build_multivariate_statistics_tools(df: pd.DataFrame,) -> list[Any]:
    """
    Create all multivariate statistical LangChain tools
    for a DataFrame.
    """
    engine = MultivariateStatistics(df)
    engine.logger.info("Building multivariate statistics tool collection.")
    tools = engine.build_tools()
    engine.logger.info("Built %d multivariate statistics tools.",len(tools),)
    return tools


def build_multivariate_visualization_tools(df: pd.DataFrame,) -> list[Any]:
    """
    Build multivariate visualization tools.
    """
    engine = MultivariateVisualization(df)
    engine.logger.info("Building multivariate visualization tool collection.")
    tools = engine.build_tools()
    engine.logger.info("Built %d multivariate visualization tools.",len(tools),)
    return tools


# ====================================================
# Build Tools
# ====================================================

def build_multivariate_tools(df: pd.DataFrame,) -> list[Any]:
    """
    Build all multivariate statistical and visualization tools.
    """
    tools: list[Any] = []
    tools.extend(build_multivariate_statistics_tools(df))
    tools.extend(build_multivariate_visualization_tools(df))
    return tools