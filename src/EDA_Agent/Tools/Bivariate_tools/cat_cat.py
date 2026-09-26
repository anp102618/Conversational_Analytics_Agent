# ====================================================
# Categorical-Categorical Bivariate Statistics
# ====================================================

"""
Categorical-Categorical Bivariate Exploratory Data Analysis.
Deterministic statistical analysis between two categorical features:
- Contingency tables
- Frequency analysis
- Row percentages
- Column percentages
- Joint percentages
- Category association
- Chi-square test
- Cramer's V
- Expected frequencies
- Standardized residuals
- Category dominance
- Conditional distributions
"""
from __future__ import annotations
from typing import Any
import numpy as np
import pandas as pd
from langchain_core.tools import tool
from scipy.stats import chi2_contingency
from src.Utils.exception_handler import CustomException
from src.Utils.logger_setup import get_log, track_performance

class CategoricalCategoricalBivariateStatistics:
    """
    Deterministic categorical-categorical bivariate statistical analyzer.
    Parameters
    ----------
    df:
        Source pandas DataFrame.
    Notes
    -----
    The DataFrame remains internal to the analyzer. LangChain tools expose
    only column names and analysis parameters.
    """

    def __init__(self, df: pd.DataFrame) -> None:
        """
        Initialize the categorical-categorical analyzer.
        Parameters
        ----------
        df:
            Source DataFrame.
        Raises
        ------
        CustomException
            If the input is not a valid DataFrame.
        """
        if not isinstance(df, pd.DataFrame):
            raise CustomException("Input must be a pandas DataFrame.")
        if df.empty:
            raise CustomException("Input DataFrame cannot be empty.")
        self.df = df
        self.logger = get_log(__name__)

    def _validate_columns(self, column_1: str, column_2: str) -> None:
        """
        Validate categorical column names and data types.
        """
        if column_1 not in self.df.columns:
            raise CustomException(f"Column '{column_1}' does not exist.")
        if column_2 not in self.df.columns:
            raise CustomException(f"Column '{column_2}' does not exist.")
        if column_1 == column_2:
            raise CustomException(
                "Categorical-categorical analysis requires two different columns."
            )
        if pd.api.types.is_numeric_dtype(self.df[column_1]):
            raise CustomException(f"Column '{column_1}' must be categorical.")
        if pd.api.types.is_numeric_dtype(self.df[column_2]):
            raise CustomException(f"Column '{column_2}' must be categorical.")

    def _get_pair(self, column_1: str, column_2: str) -> pd.DataFrame:
        """
        Return pairwise non-null categorical observations.
        """
        self._validate_columns(column_1, column_2)
        data = self.df[[column_1, column_2]].dropna()
        if data.empty:
            raise CustomException(
                "No valid observations remain after removing missing values."
            )
        return data

    def _get_contingency_table(self, column_1: str, column_2: str) -> pd.DataFrame:
        """
        Create a categorical contingency table.
        """
        data = self._get_pair(column_1, column_2)
        table = pd.crosstab(data[column_1], data[column_2], dropna=False)
        if table.empty:
            raise CustomException("Unable to create a contingency table.")
        return table

    @track_performance
    def contingency_table(self, column_1: str, column_2: str) -> dict[str, Any]:
        """
        Calculate the contingency table between two categorical features.
        """
        table = self._get_contingency_table(column_1, column_2)
        return {
            "column_1": column_1,
            "column_2": column_2,
            "rows": list(table.index.astype(str)),
            "columns": list(table.columns.astype(str)),
            "counts": table.values.tolist(),
            "total_observations": int(table.values.sum()),
        }

    @track_performance
    def row_percentage_distribution(
        self, column_1: str, column_2: str
    ) -> dict[str, Any]:
        """
        Calculate column-2 percentage distribution within each
        category of column-1.
        """
        table = self._get_contingency_table(column_1, column_2)
        percentages = table.div(table.sum(axis=1), axis=0) * 100
        percentages = percentages.fillna(0)
        return {
            "column_1": column_1,
            "column_2": column_2,
            "rows": list(percentages.index.astype(str)),
            "columns": list(percentages.columns.astype(str)),
            "percentages": percentages.round(4).values.tolist(),
        }

    @track_performance
    def column_percentage_distribution(
        self, column_1: str, column_2: str
    ) -> dict[str, Any]:
        """
        Calculate column-1 percentage distribution within each
        category of column-2.
        """
        table = self._get_contingency_table(column_1, column_2)
        percentages = table.div(table.sum(axis=0), axis=1) * 100
        percentages = percentages.fillna(0)
        return {
            "column_1": column_1,
            "column_2": column_2,
            "rows": list(percentages.index.astype(str)),
            "columns": list(percentages.columns.astype(str)),
            "percentages": percentages.round(4).values.tolist(),
        }

    @track_performance
    def joint_percentage_distribution(
        self, column_1: str, column_2: str
    ) -> dict[str, Any]:
        """
        Calculate joint percentage distribution of category pairs.
        """
        table = self._get_contingency_table(column_1, column_2)
        total = table.values.sum()
        if total == 0:
            raise CustomException(
                "Cannot calculate joint percentages for zero observations."
            )
        percentages = (table / total) * 100
        return {
            "column_1": column_1,
            "column_2": column_2,
            "rows": list(percentages.index.astype(str)),
            "columns": list(percentages.columns.astype(str)),
            "percentages": percentages.round(4).values.tolist(),
        }

    @track_performance
    def chi_square_test(self, column_1: str, column_2: str) -> dict[str, Any]:
        """
        Perform the Pearson chi-square test of independence.
        Returns
        -------
        dict
            Chi-square statistic, p-value, degrees of freedom,
            and expected frequencies.
        """
        table = self._get_contingency_table(column_1, column_2)
        if table.shape[0] < 2 or table.shape[1] < 2:
            raise CustomException(
                "Chi-square test requires at least two categories "
                "in each variable."
            )
        statistic, p_value, degrees_of_freedom, expected = chi2_contingency(table)
        expected_df = pd.DataFrame(
            expected, index=table.index, columns=table.columns
        )
        return {
            "column_1": column_1,
            "column_2": column_2,
            "chi_square_statistic": float(statistic),
            "p_value": float(p_value),
            "degrees_of_freedom": int(degrees_of_freedom),
            "expected_frequencies": expected_df.round(4).values.tolist(),
            "significant_at_0_05": bool(p_value < 0.05),
        }

    @track_performance
    def cramers_v(self, column_1: str, column_2: str) -> dict[str, Any]:
        """
        Calculate Cramer's V association strength.
        Cramer's V ranges from 0 to 1, where values closer to 0
        indicate weaker association and values closer to 1 indicate
        stronger association.
        """
        table = self._get_contingency_table(column_1, column_2)
        if table.shape[0] < 2 or table.shape[1] < 2:
            raise CustomException(
                "Cramer's V requires at least two categories in each variable."
            )
        chi_square, _, _, _ = chi2_contingency(table)
        n = table.values.sum()
        rows, columns = table.shape
        if n == 0:
            raise CustomException(
                "Cannot calculate Cramer's V for zero observations."
            )
        minimum_dimension = min(rows - 1, columns - 1)
        if minimum_dimension <= 0:
            raise CustomException("Invalid contingency table dimensions.")
        value = np.sqrt(chi_square / (n * minimum_dimension))
        return {
            "column_1": column_1,
            "column_2": column_2,
            "cramers_v": float(value),
            "sample_size": int(n),
        }

    @track_performance
    def expected_frequency_analysis(
        self, column_1: str, column_2: str
    ) -> dict[str, Any]:
        """
        Compare observed and expected frequencies under independence.
        """
        table = self._get_contingency_table(column_1, column_2)
        if table.shape[0] < 2 or table.shape[1] < 2:
            raise CustomException(
                "Expected frequency analysis requires at least "
                "two categories in each variable."
            )
        chi_square, p_value, degrees_of_freedom, expected = chi2_contingency(table)
        expected_df = pd.DataFrame(
            expected, index=table.index, columns=table.columns
        )
        residuals = table - expected_df
        return {
            "column_1": column_1,
            "column_2": column_2,
            "observed": table.values.tolist(),
            "expected": expected_df.round(4).values.tolist(),
            "residuals": residuals.round(4).values.tolist(),
            "chi_square_statistic": float(chi_square),
            "p_value": float(p_value),
            "degrees_of_freedom": int(degrees_of_freedom),
        }

    @track_performance
    def standardized_residuals(
        self, column_1: str, column_2: str
    ) -> dict[str, Any]:
        """
        Calculate standardized residuals for each category combination.
        Large positive residuals indicate combinations occurring more
        frequently than expected under independence.
        """
        table = self._get_contingency_table(column_1, column_2)
        if table.shape[0] < 2 or table.shape[1] < 2:
            raise CustomException(
                "Standardized residual analysis requires at least "
                "two categories in each variable."
            )
        _, _, _, expected = chi2_contingency(table)
        observed = table.to_numpy(dtype=float)
        expected = np.asarray(expected, dtype=float)
        total = observed.sum()
        row_totals = observed.sum(axis=1)
        column_totals = observed.sum(axis=0)
        row_proportions = row_totals / total
        column_proportions = column_totals / total
        denominator = np.sqrt(
            expected
            * (1 - row_proportions[:, np.newaxis])
            * (1 - column_proportions[np.newaxis, :])
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            residuals = np.divide(
                observed - expected,
                denominator,
                out=np.zeros_like(observed),
                where=denominator != 0,
            )
        residual_df = pd.DataFrame(
            residuals, index=table.index, columns=table.columns
        )
        return {
            "column_1": column_1,
            "column_2": column_2,
            "rows": list(residual_df.index.astype(str)),
            "columns": list(residual_df.columns.astype(str)),
            "standardized_residuals": residual_df.round(4).values.tolist(),
        }

    @track_performance
    def category_pair_ranking(
        self, column_1: str, column_2: str, top_n: int = 10
    ) -> dict[str, Any]:
        """
        Rank the most frequent category combinations.
        """
        if top_n <= 0:
            raise CustomException("top_n must be greater than zero.")
        data = self._get_pair(column_1, column_2)
        pairs = (
            data.groupby([column_1, column_2], observed=True)
            .size()
            .reset_index(name="count")
            .sort_values("count", ascending=False)
            .head(top_n)
        )
        total = len(data)
        pairs["percentage"] = pairs["count"] / total * 100
        return {
            "column_1": column_1,
            "column_2": column_2,
            "top_n": int(top_n),
            "category_pairs": pairs.to_dict(orient="records"),
        }

    @track_performance
    def conditional_distribution(
        self, column_1: str, column_2: str, category: str
    ) -> dict[str, Any]:
        """
        Calculate the distribution of column-2 categories conditioned
        on a specific category of column-1.
        """
        data = self._get_pair(column_1, column_2)
        category_values = data[column_1].astype(str)
        if category not in set(category_values):
            raise CustomException(
                f"Category '{category}' does not exist in column '{column_1}'."
            )
        subset = data[category_values == category]
        distribution = (
            subset[column_2].astype(str).value_counts(normalize=True).mul(100).round(4)
        )
        counts = subset[column_2].astype(str).value_counts()
        result = [
            {
                "category": value,
                "count": int(counts[value]),
                "percentage": float(distribution[value]),
            }
            for value in distribution.index
        ]
        return {
            "column_1": column_1,
            "column_2": column_2,
            "condition": category,
            "observations": int(len(subset)),
            "distribution": result,
        }

    def build_tools(self) -> list[Any]:
        """
        Build LangChain tools for categorical-categorical
        bivariate statistical analysis.
        Returns
        -------
        list[Any]
            List of LangChain tools.
        """
        @tool
        def contingency_table(column_1: str, column_2: str) -> dict[str, Any]:
            """
            Calculate the contingency table between two categorical
            columns.
            """
            return self.contingency_table(column_1, column_2)

        @tool
        def row_percentage_distribution(
            column_1: str, column_2: str
        ) -> dict[str, Any]:
            """
            Calculate row-wise percentage distribution between two
            categorical columns.
            """
            return self.row_percentage_distribution(column_1, column_2)

        @tool
        def column_percentage_distribution(
            column_1: str, column_2: str
        ) -> dict[str, Any]:
            """
            Calculate column-wise percentage distribution between two
            categorical columns.
            """
            return self.column_percentage_distribution(column_1, column_2)

        @tool
        def joint_percentage_distribution(
            column_1: str, column_2: str
        ) -> dict[str, Any]:
            """
            Calculate joint percentage distribution between two
            categorical columns.
            """
            return self.joint_percentage_distribution(column_1, column_2)

        @tool
        def chi_square_test(column_1: str, column_2: str) -> dict[str, Any]:
            """
            Perform a Pearson chi-square test of independence between
            two categorical columns.
            """
            return self.chi_square_test(column_1, column_2)

        @tool
        def cramers_v(column_1: str, column_2: str) -> dict[str, Any]:
            """
            Calculate Cramer's V association strength between two
            categorical columns.
            """
            return self.cramers_v(column_1, column_2)

        @tool
        def expected_frequency_analysis(
            column_1: str, column_2: str
        ) -> dict[str, Any]:
            """
            Compare observed and expected frequencies between two
            categorical columns.
            """
            return self.expected_frequency_analysis(column_1, column_2)

        @tool
        def standardized_residuals(
            column_1: str, column_2: str
        ) -> dict[str, Any]:
            """
            Calculate standardized residuals for categorical
            combinations.
            """
            return self.standardized_residuals(column_1, column_2)

        @tool
        def category_pair_ranking(
            column_1: str, column_2: str, top_n: int = 10
        ) -> dict[str, Any]:
            """
            Rank the most frequent category combinations.
            """
            return self.category_pair_ranking(column_1, column_2, top_n)

        @tool
        def conditional_distribution(
            column_1: str, column_2: str, category: str
        ) -> dict[str, Any]:
            """
            Calculate the distribution of column-2 categories
            conditioned on a category of column-1.
            """
            return self.conditional_distribution(column_1, column_2, category)

        tools = [
            contingency_table,
            row_percentage_distribution,
            column_percentage_distribution,
            joint_percentage_distribution,
            chi_square_test,
            cramers_v,
            expected_frequency_analysis,
            standardized_residuals,
            category_pair_ranking,
            conditional_distribution,
        ]
        self.logger.info(
            "Built %d categorical-categorical bivariate statistical tools.",
            len(tools),
        )
        return tools

#=================================================================

# ====================================================
# Categorical-Categorical Bivariate Statistics
# ====================================================

"""
Categorical-Categorical Bivariate Visualization.
Deterministic visual analysis between two categorical features using
Seaborn and Matplotlib.
Visualizations:
- Grouped count plot
- Stacked count plot
- Percentage stacked bar plot
- Contingency heatmap
- Row percentage heatmap
- Column percentage heatmap
- Category-pair frequency plot
"""
from __future__ import annotations
from typing import Any
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from langchain_core.tools import tool
from src.Utils.exception_handler import CustomException
from src.Utils.logger_setup import get_log, track_performance

class CategoricalCategoricalBivariateVisualization:
    """
    Deterministic categorical-categorical bivariate visualization engine.
    Parameters
    ----------
    df:
        Source pandas DataFrame.
    Notes
    -----
    The DataFrame remains internal to the visualization engine.
    LangChain tools expose only column names and visualization parameters.
    """
    DEFAULT_TOP_N = 15
    DEFAULT_FIGSIZE = (12, 6)

    def __init__(self, df: pd.DataFrame) -> None:
        """
        Initialize categorical-categorical visualization engine.
        Parameters
        ----------
        df:
            Source DataFrame.
        Raises
        ------
        CustomException
            If the DataFrame is invalid.
        """
        if not isinstance(df, pd.DataFrame):
            raise CustomException("Input must be a pandas DataFrame.")
        if df.empty:
            raise CustomException("Input DataFrame cannot be empty.")
        self.df = df
        self.logger = get_log(__name__)

    def _validate_columns(self, column_1: str, column_2: str) -> None:
        """
        Validate categorical columns.
        """
        if column_1 not in self.df.columns:
            raise CustomException(f"Column '{column_1}' does not exist.")
        if column_2 not in self.df.columns:
            raise CustomException(f"Column '{column_2}' does not exist.")
        if column_1 == column_2:
            raise CustomException("Two different categorical columns are required.")
        if pd.api.types.is_numeric_dtype(self.df[column_1]):
            raise CustomException(f"Column '{column_1}' must be categorical.")
        if pd.api.types.is_numeric_dtype(self.df[column_2]):
            raise CustomException(f"Column '{column_2}' must be categorical.")

    def _get_pair(
        self, column_1: str, column_2: str, top_n: int | None = None
    ) -> pd.DataFrame:
        """
        Return valid categorical pairs.
        Optionally restrict both columns to their most frequent
        categories to prevent unreadable visualizations.
        """
        self._validate_columns(column_1, column_2)
        data = self.df[[column_1, column_2]].dropna()
        if data.empty:
            raise CustomException(
                "No valid observations remain after removing missing values."
            )
        if top_n is not None:
            if top_n <= 0:
                raise CustomException("top_n must be greater than zero.")
            top_values_1 = data[column_1].value_counts().head(top_n).index
            top_values_2 = data[column_2].value_counts().head(top_n).index
            data = data[
                data[column_1].isin(top_values_1) & data[column_2].isin(top_values_2)
            ]
        if data.empty:
            raise CustomException("No observations remain after category filtering.")
        return data

    def _get_contingency_table(
        self, column_1: str, column_2: str, top_n: int | None = None
    ) -> pd.DataFrame:
        """
        Create a contingency table for visualization.
        """
        data = self._get_pair(column_1, column_2, top_n)
        table = pd.crosstab(data[column_1], data[column_2])
        if table.empty:
            raise CustomException("Unable to create contingency table.")
        return table

    def _create_figure(
        self, figsize: tuple[int, int] | None = None
    ) -> tuple[plt.Figure, plt.Axes]:
        """
        Create a Matplotlib figure and axes.
        """
        return plt.subplots(figsize=figsize or self.DEFAULT_FIGSIZE)

    def _format_axis(
        self, ax: plt.Axes, title: str, xlabel: str, ylabel: str
    ) -> None:
        """
        Apply common axis formatting.
        """
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=45)
        plt.tight_layout()

    @track_performance
    def grouped_count_plot(
        self,
        column_1: str,
        column_2: str,
        top_n: int = DEFAULT_TOP_N,
        figsize: tuple[int, int] | None = None,
    ) -> plt.Figure:
        """
        Create a grouped count plot for two categorical variables.
        Parameters
        ----------
        column_1:
            First categorical column.
        column_2:
            Second categorical column.
        top_n:
            Maximum number of categories retained from each column.
        figsize:
            Figure dimensions.
        """
        data = self._get_pair(column_1, column_2, top_n)
        fig, ax = self._create_figure(figsize)
        sns.countplot(data=data, x=column_1, hue=column_2, ax=ax)
        self._format_axis(
            ax, f"{column_2} Distribution by {column_1}", column_1, "Count"
        )
        return fig

    @track_performance
    def stacked_count_plot(
        self,
        column_1: str,
        column_2: str,
        top_n: int = DEFAULT_TOP_N,
        figsize: tuple[int, int] | None = None,
    ) -> plt.Figure:
        """
        Create a stacked count bar plot.
        """
        table = self._get_contingency_table(column_1, column_2, top_n)
        fig, ax = self._create_figure(figsize)
        table.plot(kind="bar", stacked=True, ax=ax)
        self._format_axis(
            ax, f"Stacked Counts: {column_1} vs {column_2}", column_1, "Count"
        )
        ax.legend(title=column_2, bbox_to_anchor=(1.02, 1), loc="upper left")
        plt.tight_layout()
        return fig

    @track_performance
    def percentage_stacked_bar_plot(
        self,
        column_1: str,
        column_2: str,
        top_n: int = DEFAULT_TOP_N,
        figsize: tuple[int, int] | None = None,
    ) -> plt.Figure:
        """
        Create a 100% stacked bar plot showing conditional percentages.
        """
        table = self._get_contingency_table(column_1, column_2, top_n)
        percentages = table.div(table.sum(axis=1), axis=0) * 100
        percentages = percentages.fillna(0)
        fig, ax = self._create_figure(figsize)
        percentages.plot(kind="bar", stacked=True, ax=ax)
        self._format_axis(
            ax,
            f"Percentage Distribution of {column_2} by {column_1}",
            column_1,
            "Percentage",
        )
        ax.set_ylim(0, 100)
        ax.legend(title=column_2, bbox_to_anchor=(1.02, 1), loc="upper left")
        plt.tight_layout()
        return fig

    @track_performance
    def contingency_heatmap(
        self,
        column_1: str,
        column_2: str,
        top_n: int = DEFAULT_TOP_N,
        figsize: tuple[int, int] | None = None,
        annot: bool = True,
    ) -> plt.Figure:
        """
        Create a heatmap of category-pair counts.
        """
        table = self._get_contingency_table(column_1, column_2, top_n)
        fig, ax = self._create_figure(figsize)
        sns.heatmap(table, annot=annot, fmt="d", linewidths=0.5, ax=ax)
        self._format_axis(
            ax,
            f"Category Frequency Heatmap: {column_1} vs {column_2}",
            column_2,
            column_1,
        )
        return fig

    @track_performance
    def row_percentage_heatmap(
        self,
        column_1: str,
        column_2: str,
        top_n: int = DEFAULT_TOP_N,
        figsize: tuple[int, int] | None = None,
        annot: bool = True,
    ) -> plt.Figure:
        """
        Create a heatmap showing column-2 percentages within
        each category of column-1.
        """
        table = self._get_contingency_table(column_1, column_2, top_n)
        percentages = table.div(table.sum(axis=1), axis=0) * 100
        percentages = percentages.fillna(0)
        fig, ax = self._create_figure(figsize)
        sns.heatmap(percentages, annot=annot, fmt=".1f", linewidths=0.5, ax=ax)
        self._format_axis(
            ax,
            f"Row Percentage Heatmap: {column_1} vs {column_2}",
            column_2,
            column_1,
        )
        return fig

    @track_performance
    def column_percentage_heatmap(
        self,
        column_1: str,
        column_2: str,
        top_n: int = DEFAULT_TOP_N,
        figsize: tuple[int, int] | None = None,
        annot: bool = True,
    ) -> plt.Figure:
        """
        Create a heatmap showing column-1 percentages within
        each category of column-2.
        """
        table = self._get_contingency_table(column_1, column_2, top_n)
        percentages = table.div(table.sum(axis=0), axis=1) * 100
        percentages = percentages.fillna(0)
        fig, ax = self._create_figure(figsize)
        sns.heatmap(percentages, annot=annot, fmt=".1f", linewidths=0.5, ax=ax)
        self._format_axis(
            ax,
            f"Column Percentage Heatmap: {column_1} vs {column_2}",
            column_2,
            column_1,
        )
        return fig

    @track_performance
    def category_pair_frequency_plot(
        self,
        column_1: str,
        column_2: str,
        top_n: int = 15,
        figsize: tuple[int, int] | None = None,
    ) -> plt.Figure:
        """
        Plot the most frequent category combinations.
        """
        data = self._get_pair(column_1, column_2)
        pair_counts = (
            data.groupby([column_1, column_2], observed=True)
            .size()
            .reset_index(name="count")
            .sort_values("count", ascending=False)
            .head(top_n)
        )
        pair_counts["pair"] = (
            pair_counts[column_1].astype(str)
            + " | "
            + pair_counts[column_2].astype(str)
        )
        fig, ax = self._create_figure(figsize)
        sns.barplot(data=pair_counts, x="count", y="pair", ax=ax)
        ax.set_title(f"Top Category Pairs: {column_1} vs {column_2}")
        ax.set_xlabel("Count")
        ax.set_ylabel(f"{column_1} | {column_2}")
        plt.tight_layout()
        return fig

    @track_performance
    def mosaic_plot(
        self,
        column_1: str,
        column_2: str,
        top_n: int = DEFAULT_TOP_N,
        figsize: tuple[int, int] | None = None,
    ) -> plt.Figure:
        """
        Create a mosaic-style visualization using a normalized
        contingency table.
        Width represents the relative frequency of column-1 categories
        while segment height represents the conditional distribution
        of column-2.
        """
        table = self._get_contingency_table(column_1, column_2, top_n)
        row_totals = table.sum(axis=1)
        row_proportions = row_totals / row_totals.sum()
        conditional = table.div(row_totals, axis=0).fillna(0)
        fig, ax = self._create_figure(figsize)
        left = 0.0
        for row_category in table.index:
            width = float(row_proportions[row_category])
            bottom = 0.0
            for col_category in table.columns:
                height = float(conditional.loc[row_category, col_category])
                if height > 0:
                    ax.bar(
                        left + width / 2,
                        height,
                        width=width,
                        bottom=bottom,
                        align="center",
                        label=str(col_category) if left == 0 and bottom == 0 else None,
                    )
                bottom += height
            ax.text(
                left + width / 2,
                -0.04,
                str(row_category),
                ha="center",
                va="top",
                transform=ax.get_xaxis_transform(),
            )
            left += width
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(f"Mosaic Plot: {column_1} vs {column_2}")
        ax.set_xlabel(column_1)
        ax.set_ylabel(f"Conditional distribution of {column_2}")
        ax.legend(title=column_2, bbox_to_anchor=(1.02, 1), loc="upper left")
        plt.tight_layout()
        return fig

    def build_tools(self) -> list[Any]:
        """
        Build LangChain visualization tools.
        Returns
        -------
        list[Any]
            List of categorical-categorical visualization tools.
        """
        @tool
        def grouped_count_plot(
            column_1: str, column_2: str, top_n: int = 15
        ) -> plt.Figure:
            """
            Create a grouped count plot comparing two categorical
            columns.
            """
            return self.grouped_count_plot(column_1, column_2, top_n)

        @tool
        def stacked_count_plot(
            column_1: str, column_2: str, top_n: int = 15
        ) -> plt.Figure:
            """
            Create a stacked count plot comparing two categorical
            columns.
            """
            return self.stacked_count_plot(column_1, column_2, top_n)

        @tool
        def percentage_stacked_bar_plot(
            column_1: str, column_2: str, top_n: int = 15
        ) -> plt.Figure:
            """
            Create a 100 percent stacked bar plot for two categorical
            columns.
            """
            return self.percentage_stacked_bar_plot(column_1, column_2, top_n)

        @tool
        def contingency_heatmap(
            column_1: str, column_2: str, top_n: int = 15, annot: bool = True
        ) -> plt.Figure:
            """
            Create a heatmap of category-pair counts.
            """
            return self.contingency_heatmap(column_1, column_2, top_n, annot=annot)

        @tool
        def row_percentage_heatmap(
            column_1: str, column_2: str, top_n: int = 15, annot: bool = True
        ) -> plt.Figure:
            """
            Create a row-percentage heatmap for two categorical
            columns.
            """
            return self.row_percentage_heatmap(column_1, column_2, top_n, annot=annot)

        @tool
        def column_percentage_heatmap(
            column_1: str, column_2: str, top_n: int = 15, annot: bool = True
        ) -> plt.Figure:
            """
            Create a column-percentage heatmap for two categorical
            columns.
            """
            return self.column_percentage_heatmap(
                column_1, column_2, top_n, annot=annot
            )

        @tool
        def category_pair_frequency_plot(
            column_1: str, column_2: str, top_n: int = 15
        ) -> plt.Figure:
            """
            Plot the most frequent category combinations.
            """
            return self.category_pair_frequency_plot(column_1, column_2, top_n)

        @tool
        def mosaic_plot(
            column_1: str, column_2: str, top_n: int = 15
        ) -> plt.Figure:
            """
            Create a mosaic-style visualization showing the relationship
            between two categorical columns.
            """
            return self.mosaic_plot(column_1, column_2, top_n)

        tools = [
            grouped_count_plot,
            stacked_count_plot,
            percentage_stacked_bar_plot,
            contingency_heatmap,
            row_percentage_heatmap,
            column_percentage_heatmap,
            category_pair_frequency_plot,
            mosaic_plot,
        ]
        self.logger.info(
            "Built %d categorical-categorical bivariate visualization tools.",
            len(tools),
        )
        return tools


#=========================================================================

# ====================================================
# Factory
# ====================================================
def build_categorical_categorical_bivariate_statistics_tools(df: pd.DataFrame,) -> list[Any]:
    """
    Create all categorical-categorical bivariate statistical LangChain
    tools for a DataFrame.
    Parameters
    ----------
    df:
        DataFrame that the tools will analyze.
    Returns
    -------
    list[Any]
        LangChain tools covering categorical-categorical bivariate
        analysis.
    Raises
    ------
    CustomException
        If the DataFrame is invalid.
    """
    engine = CategoricalCategoricalBivariateStatistics(df)
    engine.logger.info("Building categorical-categorical bivariate statistics tool collection.")
    tools = engine.build_tools()
    engine.logger.info("Built %d categorical-categorical bivariate statistics tools.", len(tools))
    return tools

def build_categorical_categorical_bivariate_visualization_tools(df: pd.DataFrame,) -> list[Any]:
    """
    Build categorical-categorical bivariate visualization tools.
    Parameters
    ----------
    df:
        Source DataFrame.
    Returns
    -------
    list[Any]
        LangChain visualization tools.
    Raises
    ------
    CustomException
        If the DataFrame is invalid.
    """
    engine = CategoricalCategoricalBivariateVisualization(df)
    engine.logger.info("Building categorical-categorical bivariate visualization tool collection.")
    tools = engine.build_tools()
    engine.logger.info("Built %d categorical-categorical bivariate visualization tools.", len(tools))
    return tools

# ====================================================
# Build Tools
# ====================================================
def build_categorical_categorical_bivariate_tools(df: pd.DataFrame) -> list[Any]:
    """
    Build all categorical-categorical bivariate tools.
    Includes
    --------
    - Categorical-categorical bivariate statistics
    - Categorical-categorical bivariate visualizations
    Parameters
    ----------
    df:
        Source DataFrame.
    Returns
    -------
    list[Any]
        Complete categorical-categorical bivariate tool collection.
    """
    tools: list[Any] = []
    tools.extend(build_categorical_categorical_bivariate_statistics_tools(df))
    tools.extend(build_categorical_categorical_bivariate_visualization_tools(df))
    return tools
