#====================================================
# Categorical-Datetime Bivariate Statistics
#====================================================

"""
Categorical-Datetime Bivariate Exploratory Data Analysis.
Deterministic temporal analysis between categorical and datetime features:
- Temporal category frequencies
- Temporal category percentages
- Category distribution by period
- Category activity over time
- Category dominance by period
- Category presence across periods
- Category changes over time
- Missing values by period
"""
from __future__ import annotations
from typing import Any
import numpy as np
import pandas as pd
from langchain_core.tools import tool
from src.Utils.exception_handler import CustomException
from src.Utils.logger_setup import get_log, track_performance

class CategoricalDatetimeBivariateStatistics:
    """
    Deterministic categorical-datetime bivariate statistical analysis.
    The class analyzes the relationship between a categorical feature
    and a datetime feature across temporal periods.
    Parameters
    ----------
    df:
        Source pandas DataFrame.
    """
    DEFAULT_FREQUENCY = "M"
    DEFAULT_TOP_N = 5

    def __init__(self, df: pd.DataFrame) -> None:
        """
        Initialize categorical-datetime bivariate statistics engine.
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
            raise CustomException("Expected a pandas DataFrame.")
        if df.empty:
            raise CustomException(
                "Cannot initialize analysis with an empty DataFrame."
            )
        self.df = df
        self.logger = get_log(__name__)

    def _validate_columns(
        self, categorical_column: str, datetime_column: str
    ) -> None:
        """
        Validate categorical and datetime columns.
        Parameters
        ----------
        categorical_column:
            Name of categorical column.
        datetime_column:
            Name of datetime column.
        Raises
        ------
        CustomException
            If columns are invalid.
        """
        if categorical_column not in self.df.columns:
            raise CustomException(
                f"Categorical column '{categorical_column}' does not exist."
            )
        if datetime_column not in self.df.columns:
            raise CustomException(
                f"Datetime column '{datetime_column}' does not exist."
            )
        if categorical_column == datetime_column:
            raise CustomException(
                "Categorical and datetime columns must be different."
            )
        if pd.api.types.is_numeric_dtype(self.df[categorical_column]):
            raise CustomException(
                f"Column '{categorical_column}' must be categorical."
            )
        if not pd.api.types.is_datetime64_any_dtype(self.df[datetime_column]):
            raise CustomException(
                f"Column '{datetime_column}' must have a datetime dtype."
            )

    def _get_pair(
        self, categorical_column: str, datetime_column: str
    ) -> pd.DataFrame:
        """
        Return a cleaned categorical-datetime pair.
        Missing categorical or datetime observations are removed.
        Parameters
        ----------
        categorical_column:
            Name of categorical column.
        datetime_column:
            Name of datetime column.
        Returns
        -------
        pd.DataFrame
            Cleaned pair DataFrame.
        """
        self._validate_columns(categorical_column, datetime_column)
        data = self.df[[categorical_column, datetime_column]].dropna(
            subset=[categorical_column, datetime_column]
        )
        if data.empty:
            raise CustomException(
                "No valid observations remain after removing "
                "missing categorical and datetime values."
            )
        return data

    def _get_temporal_data(
        self,
        categorical_column: str,
        datetime_column: str,
        frequency: str,
    ) -> pd.DataFrame:
        """
        Prepare categorical-datetime data with temporal periods.
        """
        data = self._get_pair(categorical_column, datetime_column)
        try:
            data["_period"] = data[datetime_column].dt.to_period(frequency)
        except Exception as exc:
            raise CustomException(
                f"Invalid temporal frequency '{frequency}'."
            ) from exc
        return data

    @track_performance
    def temporal_category_frequency(
        self,
        categorical_column: str,
        datetime_column: str,
        frequency: str = DEFAULT_FREQUENCY,
    ) -> dict[str, Any]:
        """
        Calculate category frequencies across temporal periods.
        Parameters
        ----------
        categorical_column:
            Categorical feature.
        datetime_column:
            Datetime feature.
        frequency:
            Pandas period frequency such as M, Q, W, or Y.
        Returns
        -------
        dict[str, Any]
            Category counts for each temporal period.
        """
        data = self._get_temporal_data(
            categorical_column, datetime_column, frequency
        )
        counts = pd.crosstab(data["_period"], data[categorical_column])
        return {
            "categorical_column": categorical_column,
            "datetime_column": datetime_column,
            "frequency": frequency,
            "periods": [str(value) for value in counts.index],
            "category_frequency": counts.to_dict(),
        }

    @track_performance
    def temporal_category_percentage(
        self,
        categorical_column: str,
        datetime_column: str,
        frequency: str = DEFAULT_FREQUENCY,
    ) -> dict[str, Any]:
        """
        Calculate category percentages within each time period.
        Parameters
        ----------
        categorical_column:
            Categorical feature.
        datetime_column:
            Datetime feature.
        frequency:
            Pandas period frequency.
        Returns
        -------
        dict[str, Any]
            Percentage distribution of categories by period.
        """
        data = self._get_temporal_data(
            categorical_column, datetime_column, frequency
        )
        counts = pd.crosstab(data["_period"], data[categorical_column])
        percentages = counts.div(counts.sum(axis=1), axis=0).mul(100)
        return {
            "categorical_column": categorical_column,
            "datetime_column": datetime_column,
            "frequency": frequency,
            "percentage_distribution": percentages.round(4).to_dict(),
        }

    @track_performance
    def category_distribution_by_period(
        self,
        categorical_column: str,
        datetime_column: str,
        category: str,
        frequency: str = DEFAULT_FREQUENCY,
    ) -> dict[str, Any]:
        """
        Analyze the frequency and percentage of one category
        across temporal periods.
        Parameters
        ----------
        categorical_column:
            Categorical feature.
        datetime_column:
            Datetime feature.
        category:
            Category to analyze.
        frequency:
            Pandas period frequency.
        Returns
        -------
        dict[str, Any]
            Temporal distribution of the selected category.
        """
        data = self._get_temporal_data(
            categorical_column, datetime_column, frequency
        )
        category_data = data[
            data[categorical_column].astype(str) == str(category)
        ]
        counts = category_data.groupby("_period", observed=True).size()
        total_counts = data.groupby("_period", observed=True).size()
        percentages = counts.div(total_counts).mul(100).fillna(0)
        periods = total_counts.index
        result = pd.DataFrame({
            "count": counts.reindex(periods, fill_value=0),
            "percentage": percentages.reindex(periods, fill_value=0),
        })
        return {
            "categorical_column": categorical_column,
            "datetime_column": datetime_column,
            "category": category,
            "frequency": frequency,
            "temporal_distribution": result.round(4).to_dict(orient="index"),
        }

    @track_performance
    def top_categories_by_period(
        self,
        categorical_column: str,
        datetime_column: str,
        frequency: str = DEFAULT_FREQUENCY,
        top_n: int = DEFAULT_TOP_N,
    ) -> dict[str, Any]:
        """
        Identify the most frequent categories in each time period.
        Parameters
        ----------
        categorical_column:
            Categorical feature.
        datetime_column:
            Datetime feature.
        frequency:
            Pandas period frequency.
        top_n:
            Number of categories to return per period.
        Returns
        -------
        dict[str, Any]
            Top categories for every temporal period.
        """
        if top_n <= 0:
            raise CustomException("top_n must be greater than zero.")
        data = self._get_temporal_data(
            categorical_column, datetime_column, frequency
        )
        counts = pd.crosstab(data["_period"], data[categorical_column])
        results: dict[str, Any] = {}
        for period, row in counts.iterrows():
            top = row.sort_values(ascending=False).head(top_n)
            results[str(period)] = {
                str(category): int(count) for category, count in top.items()
            }
        return {
            "categorical_column": categorical_column,
            "datetime_column": datetime_column,
            "frequency": frequency,
            "top_n": top_n,
            "top_categories_by_period": results,
        }

    @track_performance
    def category_dominance_by_period(
        self,
        categorical_column: str,
        datetime_column: str,
        frequency: str = DEFAULT_FREQUENCY,
    ) -> dict[str, Any]:
        """
        Identify the dominant category in each temporal period.
        Dominance is determined by the highest category frequency
        within each period.
        Parameters
        ----------
        categorical_column:
            Categorical feature.
        datetime_column:
            Datetime feature.
        frequency:
            Pandas period frequency.
        Returns
        -------
        dict[str, Any]
            Dominant category and its share for each period.
        """
        data = self._get_temporal_data(
            categorical_column, datetime_column, frequency
        )
        counts = pd.crosstab(data["_period"], data[categorical_column])
        results: dict[str, Any] = {}
        for period, row in counts.iterrows():
            dominant_category = row.idxmax()
            dominant_count = int(row.max())
            total_count = int(row.sum())
            percentage = (
                dominant_count / total_count * 100 if total_count > 0 else np.nan
            )
            results[str(period)] = {
                "dominant_category": str(dominant_category),
                "count": dominant_count,
                "percentage": round(float(percentage), 4),
            }
        return {
            "categorical_column": categorical_column,
            "datetime_column": datetime_column,
            "frequency": frequency,
            "dominance_by_period": results,
        }

    @track_performance
    def category_presence_across_periods(
        self,
        categorical_column: str,
        datetime_column: str,
        frequency: str = DEFAULT_FREQUENCY,
    ) -> dict[str, Any]:
        """
        Determine how many temporal periods each category appears in.
        Parameters
        ----------
        categorical_column:
            Categorical feature.
        datetime_column:
            Datetime feature.
        frequency:
            Pandas period frequency.
        Returns
        -------
        dict[str, Any]
            Period presence statistics for each category.
        """
        data = self._get_temporal_data(
            categorical_column, datetime_column, frequency
        )
        counts = pd.crosstab(data[categorical_column], data["_period"])
        period_count = len(counts.columns)
        presence = (counts > 0).sum(axis=1)
        results = pd.DataFrame({
            "periods_present": presence,
            "total_periods": period_count,
            "presence_percentage": (
                presence.div(period_count).mul(100) if period_count > 0 else np.nan
            ),
        }).sort_values("periods_present", ascending=False)
        return {
            "categorical_column": categorical_column,
            "datetime_column": datetime_column,
            "frequency": frequency,
            "category_presence": results.round(4).to_dict(orient="index"),
        }

    @track_performance
    def category_temporal_change(
        self,
        categorical_column: str,
        datetime_column: str,
        category: str,
        frequency: str = DEFAULT_FREQUENCY,
    ) -> dict[str, Any]:
        """
        Calculate period-over-period changes for one category.
        Parameters
        ----------
        categorical_column:
            Categorical feature.
        datetime_column:
            Datetime feature.
        category:
            Category to analyze.
        frequency:
            Pandas period frequency.
        Returns
        -------
        dict[str, Any]
            Counts, absolute changes, and percentage changes.
        """
        data = self._get_temporal_data(
            categorical_column, datetime_column, frequency
        )
        category_data = data[
            data[categorical_column].astype(str) == str(category)
        ]
        counts = category_data.groupby("_period", observed=True).size()
        all_periods = pd.period_range(
            data["_period"].min(), data["_period"].max(), freq=frequency
        )
        counts = counts.reindex(all_periods, fill_value=0)
        absolute_change = counts.diff()
        percentage_change = (
            counts.pct_change(fill_method=None)
            .replace([np.inf, -np.inf], np.nan)
            .mul(100)
        )
        result = pd.DataFrame({
            "count": counts,
            "absolute_change": absolute_change,
            "percentage_change": percentage_change,
        })
        return {
            "categorical_column": categorical_column,
            "datetime_column": datetime_column,
            "category": category,
            "frequency": frequency,
            "temporal_change": result.round(4).to_dict(orient="index"),
        }

    @track_performance
    def missing_values_by_period(
        self,
        categorical_column: str,
        datetime_column: str,
        frequency: str = DEFAULT_FREQUENCY,
    ) -> dict[str, Any]:
        """
        Analyze categorical missingness across temporal periods.
        Datetime missing values are excluded because they cannot
        be assigned to a temporal period. Categorical missing
        values are intentionally preserved.
        Parameters
        ----------
        categorical_column:
            Categorical feature.
        datetime_column:
            Datetime feature.
        frequency:
            Pandas period frequency.
        Returns
        -------
        dict[str, Any]
            Missing count and missing percentage by period.
        """
        self._validate_columns(categorical_column, datetime_column)
        data = self.df[[categorical_column, datetime_column]].dropna(
            subset=[datetime_column]
        )
        if data.empty:
            raise CustomException("No valid datetime observations remain.")
        data["_period"] = data[datetime_column].dt.to_period(frequency)
        grouped = data.groupby("_period", observed=True)
        missing_count = grouped[categorical_column].apply(
            lambda series: int(series.isna().sum())
        )
        total_count = grouped.size()
        missing_percentage = missing_count.div(total_count).mul(100)
        result = pd.DataFrame({
            "total_count": total_count,
            "missing_count": missing_count,
            "missing_percentage": missing_percentage,
        })
        return {
            "categorical_column": categorical_column,
            "datetime_column": datetime_column,
            "frequency": frequency,
            "missing_values_by_period": result.round(4).to_dict(orient="index"),
        }

    @track_performance
    def category_temporal_summary(
        self,
        categorical_column: str,
        datetime_column: str,
        frequency: str = DEFAULT_FREQUENCY,
    ) -> dict[str, Any]:
        """
        Generate a compact temporal summary for each category.
        Parameters
        ----------
        categorical_column:
            Categorical feature.
        datetime_column:
            Datetime feature.
        frequency:
            Pandas period frequency.
        Returns
        -------
        dict[str, Any]
            Temporal summary for each category.
        """
        data = self._get_temporal_data(
            categorical_column, datetime_column, frequency
        )
        grouped = data.groupby(categorical_column, observed=True)["_period"]
        summary = grouped.agg(
            first_period="min", last_period="max", active_periods="nunique"
        )
        total_periods = data["_period"].nunique()
        summary["presence_percentage"] = (
            summary["active_periods"].div(total_periods).mul(100)
        )
        summary = summary.sort_values("active_periods", ascending=False)
        summary["first_period"] = summary["first_period"].astype(str)
        summary["last_period"] = summary["last_period"].astype(str)
        return {
            "categorical_column": categorical_column,
            "datetime_column": datetime_column,
            "frequency": frequency,
            "category_temporal_summary": summary.round(4).to_dict(orient="index"),
        }

    def build_tools(self) -> list[Any]:
        """
        Build LangChain tools for categorical-datetime
        bivariate statistical analysis.
        Returns
        -------
        list[Any]
            LangChain tools.
        """
        @tool
        def temporal_category_frequency(
            categorical_column: str,
            datetime_column: str,
            frequency: str = self.DEFAULT_FREQUENCY,
        ) -> dict[str, Any]:
            """
            Calculate category frequencies across temporal periods.
            """
            return self.temporal_category_frequency(
                categorical_column, datetime_column, frequency
            )

        @tool
        def temporal_category_percentage(
            categorical_column: str,
            datetime_column: str,
            frequency: str = self.DEFAULT_FREQUENCY,
        ) -> dict[str, Any]:
            """
            Calculate category percentages within each time period.
            """
            return self.temporal_category_percentage(
                categorical_column, datetime_column, frequency
            )

        @tool
        def category_distribution_by_period(
            categorical_column: str,
            datetime_column: str,
            category: str,
            frequency: str = self.DEFAULT_FREQUENCY,
        ) -> dict[str, Any]:
            """
            Analyze one category across temporal periods.
            """
            return self.category_distribution_by_period(
                categorical_column, datetime_column, category, frequency
            )

        @tool
        def top_categories_by_period(
            categorical_column: str,
            datetime_column: str,
            frequency: str = self.DEFAULT_FREQUENCY,
            top_n: int = self.DEFAULT_TOP_N,
        ) -> dict[str, Any]:
            """
            Identify the most frequent categories by period.
            """
            return self.top_categories_by_period(
                categorical_column, datetime_column, frequency, top_n
            )

        @tool
        def category_dominance_by_period(
            categorical_column: str,
            datetime_column: str,
            frequency: str = self.DEFAULT_FREQUENCY,
        ) -> dict[str, Any]:
            """
            Identify the dominant category in each time period.
            """
            return self.category_dominance_by_period(
                categorical_column, datetime_column, frequency
            )

        @tool
        def category_presence_across_periods(
            categorical_column: str,
            datetime_column: str,
            frequency: str = self.DEFAULT_FREQUENCY,
        ) -> dict[str, Any]:
            """
            Determine category presence across temporal periods.
            """
            return self.category_presence_across_periods(
                categorical_column, datetime_column, frequency
            )

        @tool
        def category_temporal_change(
            categorical_column: str,
            datetime_column: str,
            category: str,
            frequency: str = self.DEFAULT_FREQUENCY,
        ) -> dict[str, Any]:
            """
            Calculate temporal changes for one category.
            """
            return self.category_temporal_change(
                categorical_column, datetime_column, category, frequency
            )

        @tool
        def missing_values_by_period(
            categorical_column: str,
            datetime_column: str,
            frequency: str = self.DEFAULT_FREQUENCY,
        ) -> dict[str, Any]:
            """
            Analyze categorical missingness by temporal period.
            """
            return self.missing_values_by_period(
                categorical_column, datetime_column, frequency
            )

        @tool
        def category_temporal_summary(
            categorical_column: str,
            datetime_column: str,
            frequency: str = self.DEFAULT_FREQUENCY,
        ) -> dict[str, Any]:
            """
            Generate a temporal summary for each category.
            """
            return self.category_temporal_summary(
                categorical_column, datetime_column, frequency
            )

        tools: list[Any] = [
            temporal_category_frequency,
            temporal_category_percentage,
            category_distribution_by_period,
            top_categories_by_period,
            category_dominance_by_period,
            category_presence_across_periods,
            category_temporal_change,
            missing_values_by_period,
            category_temporal_summary,
        ]
        self.logger.info(
            "Built %d categorical-datetime bivariate statistics tools.",
            len(tools),
        )
        return tools

#===========================================================================
#====================================================
# Categorical-Datetime Bivariate Visualization
#====================================================

"""
Categorical-Datetime Bivariate Visualization.
Deterministic visual analysis between categorical and datetime features:
- Temporal category counts
- Temporal percentage composition
- Category trends over time
- Category distribution by period
- Temporal category heatmaps
- Category dominance over time
- Category presence across periods
- Missingness over time
"""
from __future__ import annotations
from typing import Any
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from langchain_core.tools import tool
from src.Utils.exception_handler import CustomException
from src.Utils.logger_setup import get_log, track_performance

class CategoricalDatetimeBivariateVisualization:
    """
    Visualization engine for categorical-datetime bivariate analysis.
    Parameters
    ----------
    df:
        Source pandas DataFrame.
    """
    DEFAULT_FREQUENCY = "M"
    DEFAULT_TOP_N = 10
    DEFAULT_FIGSIZE = (12, 6)

    def __init__(self, df: pd.DataFrame) -> None:
        """
        Initialize categorical-datetime visualization engine.
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
            raise CustomException("Expected a pandas DataFrame.")
        if df.empty:
            raise CustomException(
                "Cannot initialize visualization with an empty DataFrame."
            )
        self.df = df
        self.logger = get_log(__name__)

    def _validate_columns(
        self, categorical_column: str, datetime_column: str
    ) -> None:
        """
        Validate categorical and datetime columns.
        Parameters
        ----------
        categorical_column:
            Name of categorical column.
        datetime_column:
            Name of datetime column.
        Raises
        ------
        CustomException
            If columns are invalid.
        """
        if categorical_column not in self.df.columns:
            raise CustomException(
                f"Categorical column '{categorical_column}' does not exist."
            )
        if datetime_column not in self.df.columns:
            raise CustomException(
                f"Datetime column '{datetime_column}' does not exist."
            )
        if categorical_column == datetime_column:
            raise CustomException(
                "Categorical and datetime columns must be different."
            )
        if pd.api.types.is_numeric_dtype(self.df[categorical_column]):
            raise CustomException(
                f"Column '{categorical_column}' must be categorical."
            )
        if not pd.api.types.is_datetime64_any_dtype(self.df[datetime_column]):
            raise CustomException(
                f"Column '{datetime_column}' must have a datetime dtype."
            )

    def _get_pair(
        self,
        categorical_column: str,
        datetime_column: str,
        top_n: int | None = None,
    ) -> pd.DataFrame:
        """
        Prepare categorical-datetime data for visualization.
        Missing categorical and datetime observations are removed.
        Optionally keeps only the top-N most frequent categories.
        Parameters
        ----------
        categorical_column:
            Name of categorical column.
        datetime_column:
            Name of datetime column.
        top_n:
            Number of most frequent categories to retain.
        Returns
        -------
        pd.DataFrame
            Prepared DataFrame.
        """
        self._validate_columns(categorical_column, datetime_column)
        if top_n is not None and top_n <= 0:
            raise CustomException("top_n must be greater than zero.")
        data = self.df[[categorical_column, datetime_column]].dropna(
            subset=[categorical_column, datetime_column]
        )
        if data.empty:
            raise CustomException(
                "No valid observations remain after removing missing values."
            )
        if top_n is not None:
            top_categories = (
                data[categorical_column].value_counts().head(top_n).index
            )
            data = data[data[categorical_column].isin(top_categories)]
        if data.empty:
            raise CustomException(
                "No observations remain after category filtering."
            )
        return data

    def _get_temporal_data(
        self,
        categorical_column: str,
        datetime_column: str,
        frequency: str,
        top_n: int | None = None,
    ) -> pd.DataFrame:
        """
        Prepare data with a temporal period column.
        """
        data = self._get_pair(categorical_column, datetime_column, top_n)
        try:
            data["_period"] = data[datetime_column].dt.to_period(frequency)
        except Exception as exc:
            raise CustomException(
                f"Invalid temporal frequency '{frequency}'."
            ) from exc
        return data

    def _create_figure(
        self, figsize: tuple[int, int] | None = None
    ) -> tuple[plt.Figure, plt.Axes]:
        """
        Create a Matplotlib figure and axes.
        """
        return plt.subplots(figsize=figsize or self.DEFAULT_FIGSIZE)

    def _format_axis(
        self, axis: plt.Axes, title: str, xlabel: str, ylabel: str
    ) -> None:
        """
        Apply standard axis formatting.
        """
        axis.set_title(title)
        axis.set_xlabel(xlabel)
        axis.set_ylabel(ylabel)
        axis.tick_params(axis="x", rotation=45)
        axis.get_figure().tight_layout()

    @track_performance
    def temporal_category_count_plot(
        self,
        categorical_column: str,
        datetime_column: str,
        frequency: str = DEFAULT_FREQUENCY,
        top_n: int = DEFAULT_TOP_N,
    ) -> plt.Figure:
        """
        Plot category frequencies across temporal periods.
        Parameters
        ----------
        categorical_column:
            Categorical feature.
        datetime_column:
            Datetime feature.
        frequency:
            Temporal aggregation frequency.
        top_n:
            Number of categories to display.
        Returns
        -------
        plt.Figure
            Category count visualization.
        """
        data = self._get_temporal_data(
            categorical_column, datetime_column, frequency, top_n
        )
        grouped = (
            data.groupby(["_period", categorical_column], observed=True)
            .size()
            .reset_index(name="count")
        )
        figure, axis = self._create_figure()
        sns.lineplot(
            data=grouped,
            x="_period",
            y="count",
            hue=categorical_column,
            marker="o",
            ax=axis,
        )
        self._format_axis(axis, "Category Frequency Over Time", "Period", "Count")
        return figure

    @track_performance
    def temporal_percentage_plot(
        self,
        categorical_column: str,
        datetime_column: str,
        frequency: str = DEFAULT_FREQUENCY,
        top_n: int = DEFAULT_TOP_N,
    ) -> plt.Figure:
        """
        Plot category percentage composition over time.
        Parameters
        ----------
        categorical_column:
            Categorical feature.
        datetime_column:
            Datetime feature.
        frequency:
            Temporal aggregation frequency.
        top_n:
            Number of categories to display.
        Returns
        -------
        plt.Figure
            Percentage trend visualization.
        """
        data = self._get_temporal_data(
            categorical_column, datetime_column, frequency, top_n
        )
        counts = pd.crosstab(data["_period"], data[categorical_column])
        percentages = (
            counts.div(counts.sum(axis=1), axis=0).mul(100).reset_index()
        )
        melted = percentages.melt(
            id_vars="_period",
            var_name=categorical_column,
            value_name="percentage",
        )
        figure, axis = self._create_figure()
        sns.lineplot(
            data=melted,
            x="_period",
            y="percentage",
            hue=categorical_column,
            marker="o",
            ax=axis,
        )
        self._format_axis(
            axis, "Category Percentage Over Time", "Period", "Percentage"
        )
        return figure

    @track_performance
    def temporal_stacked_bar_plot(
        self,
        categorical_column: str,
        datetime_column: str,
        frequency: str = DEFAULT_FREQUENCY,
        top_n: int = DEFAULT_TOP_N,
    ) -> plt.Figure:
        """
        Plot stacked category counts across time.
        Parameters
        ----------
        categorical_column:
            Categorical feature.
        datetime_column:
            Datetime feature.
        frequency:
            Temporal aggregation frequency.
        top_n:
            Number of categories to display.
        Returns
        -------
        plt.Figure
            Stacked temporal count plot.
        """
        data = self._get_temporal_data(
            categorical_column, datetime_column, frequency, top_n
        )
        counts = pd.crosstab(data["_period"], data[categorical_column])
        figure, axis = self._create_figure()
        counts.plot(kind="bar", stacked=True, ax=axis)
        self._format_axis(
            axis, "Category Composition Over Time", "Period", "Count"
        )
        return figure

    @track_performance
    def temporal_percentage_stacked_bar_plot(
        self,
        categorical_column: str,
        datetime_column: str,
        frequency: str = DEFAULT_FREQUENCY,
        top_n: int = DEFAULT_TOP_N,
    ) -> plt.Figure:
        """
        Plot 100% stacked category percentages across time.
        Parameters
        ----------
        categorical_column:
            Categorical feature.
        datetime_column:
            Datetime feature.
        frequency:
            Temporal aggregation frequency.
        top_n:
            Number of categories to display.
        Returns
        -------
        plt.Figure
            100% stacked temporal percentage plot.
        """
        data = self._get_temporal_data(
            categorical_column, datetime_column, frequency, top_n
        )
        counts = pd.crosstab(data["_period"], data[categorical_column])
        percentages = counts.div(counts.sum(axis=1), axis=0).mul(100)
        figure, axis = self._create_figure()
        percentages.plot(kind="bar", stacked=True, ax=axis)
        self._format_axis(
            axis,
            "Category Percentage Composition Over Time",
            "Period",
            "Percentage",
        )
        return figure

    @track_performance
    def category_trend_plot(
        self,
        categorical_column: str,
        datetime_column: str,
        category: str,
        frequency: str = DEFAULT_FREQUENCY,
    ) -> plt.Figure:
        """
        Plot the frequency of one category over time.
        Parameters
        ----------
        categorical_column:
            Categorical feature.
        datetime_column:
            Datetime feature.
        category:
            Category to visualize.
        frequency:
            Temporal aggregation frequency.
        Returns
        -------
        plt.Figure
            Category temporal trend.
        """
        data = self._get_temporal_data(
            categorical_column, datetime_column, frequency
        )
        data = data[data[categorical_column].astype(str) == str(category)]
        if data.empty:
            raise CustomException(f"Category '{category}' was not found.")
        counts = (
            data.groupby("_period", observed=True)
            .size()
            .reset_index(name="count")
        )
        figure, axis = self._create_figure()
        sns.lineplot(data=counts, x="_period", y="count", marker="o", ax=axis)
        self._format_axis(
            axis, f"Temporal Trend: {category}", "Period", "Count"
        )
        return figure

    @track_performance
    def temporal_category_heatmap(
        self,
        categorical_column: str,
        datetime_column: str,
        frequency: str = DEFAULT_FREQUENCY,
        top_n: int = DEFAULT_TOP_N,
    ) -> plt.Figure:
        """
        Plot a heatmap of category frequency across time.
        Parameters
        ----------
        categorical_column:
            Categorical feature.
        datetime_column:
            Datetime feature.
        frequency:
            Temporal aggregation frequency.
        top_n:
            Number of categories to display.
        Returns
        -------
        plt.Figure
            Temporal category heatmap.
        """
        data = self._get_temporal_data(
            categorical_column, datetime_column, frequency, top_n
        )
        counts = pd.crosstab(data[categorical_column], data["_period"])
        figure, axis = self._create_figure(figsize=(14, 7))
        sns.heatmap(counts, annot=False, fmt="g", cmap="Blues", ax=axis)
        self._format_axis(
            axis, "Category Frequency by Period", "Period", "Category"
        )
        return figure

    @track_performance
    def temporal_percentage_heatmap(
        self,
        categorical_column: str,
        datetime_column: str,
        frequency: str = DEFAULT_FREQUENCY,
        top_n: int = DEFAULT_TOP_N,
    ) -> plt.Figure:
        """
        Plot category percentages across temporal periods.
        Parameters
        ----------
        categorical_column:
            Categorical feature.
        datetime_column:
            Datetime feature.
        frequency:
            Temporal aggregation frequency.
        top_n:
            Number of categories to display.
        Returns
        -------
        plt.Figure
            Temporal percentage heatmap.
        """
        data = self._get_temporal_data(
            categorical_column, datetime_column, frequency, top_n
        )
        counts = pd.crosstab(data["_period"], data[categorical_column])
        percentages = counts.div(counts.sum(axis=1), axis=0).mul(100).T
        figure, axis = self._create_figure(figsize=(14, 7))
        sns.heatmap(percentages, annot=False, fmt=".1f", cmap="Blues", ax=axis)
        self._format_axis(
            axis, "Category Percentage by Period", "Period", "Category"
        )
        return figure

    @track_performance
    def category_presence_plot(
        self,
        categorical_column: str,
        datetime_column: str,
        frequency: str = DEFAULT_FREQUENCY,
        top_n: int = DEFAULT_TOP_N,
    ) -> plt.Figure:
        """
        Visualize category presence across temporal periods.
        Parameters
        ----------
        categorical_column:
            Categorical feature.
        datetime_column:
            Datetime feature.
        frequency:
            Temporal aggregation frequency.
        top_n:
            Number of categories to display.
        Returns
        -------
        plt.Figure
            Category presence heatmap.
        """
        data = self._get_temporal_data(
            categorical_column, datetime_column, frequency, top_n
        )
        presence = (pd.crosstab(data[categorical_column], data["_period"]) > 0).astype(
            int
        )
        figure, axis = self._create_figure(figsize=(14, 7))
        sns.heatmap(presence, cmap="Blues", cbar=False, ax=axis)
        self._format_axis(
            axis, "Category Presence Across Periods", "Period", "Category"
        )
        return figure

    @track_performance
    def temporal_missingness_plot(
        self,
        categorical_column: str,
        datetime_column: str,
        frequency: str = DEFAULT_FREQUENCY,
    ) -> plt.Figure:
        """
        Plot categorical missingness rate across time.
        Datetime missing values are excluded because they cannot
        be assigned to a temporal period. Categorical missing
        values are retained.
        Parameters
        ----------
        categorical_column:
            Categorical feature.
        datetime_column:
            Datetime feature.
        frequency:
            Temporal aggregation frequency.
        Returns
        -------
        plt.Figure
            Missingness trend.
        """
        self._validate_columns(categorical_column, datetime_column)
        data = self.df[[categorical_column, datetime_column]].dropna(
            subset=[datetime_column]
        )
        if data.empty:
            raise CustomException("No valid datetime observations remain.")
        data["_period"] = data[datetime_column].dt.to_period(frequency)
        missing_rate = (
            data.groupby("_period", observed=True)[categorical_column]
            .apply(lambda series: series.isna().mean() * 100)
            .reset_index(name="missing_percentage")
        )
        figure, axis = self._create_figure()
        sns.lineplot(
            data=missing_rate,
            x="_period",
            y="missing_percentage",
            marker="o",
            ax=axis,
        )
        self._format_axis(
            axis,
            "Categorical Missingness Over Time",
            "Period",
            "Missing Percentage",
        )
        return figure

    def build_tools(self) -> list[Any]:
        """
        Build all categorical-datetime visualization tools.
        Returns
        -------
        list[Any]
            LangChain visualization tools.
        """
        @tool
        def temporal_category_count_plot(
            categorical_column: str,
            datetime_column: str,
            frequency: str = self.DEFAULT_FREQUENCY,
            top_n: int = self.DEFAULT_TOP_N,
        ) -> plt.Figure:
            """
            Plot category frequencies across temporal periods.
            """
            return self.temporal_category_count_plot(
                categorical_column, datetime_column, frequency, top_n
            )

        @tool
        def temporal_percentage_plot(
            categorical_column: str,
            datetime_column: str,
            frequency: str = self.DEFAULT_FREQUENCY,
            top_n: int = self.DEFAULT_TOP_N,
        ) -> plt.Figure:
            """
            Plot category percentage composition over time.
            """
            return self.temporal_percentage_plot(
                categorical_column, datetime_column, frequency, top_n
            )

        @tool
        def temporal_stacked_bar_plot(
            categorical_column: str,
            datetime_column: str,
            frequency: str = self.DEFAULT_FREQUENCY,
            top_n: int = self.DEFAULT_TOP_N,
        ) -> plt.Figure:
            """
            Plot stacked category counts across time.
            """
            return self.temporal_stacked_bar_plot(
                categorical_column, datetime_column, frequency, top_n
            )

        @tool
        def temporal_percentage_stacked_bar_plot(
            categorical_column: str,
            datetime_column: str,
            frequency: str = self.DEFAULT_FREQUENCY,
            top_n: int = self.DEFAULT_TOP_N,
        ) -> plt.Figure:
            """
            Plot 100% stacked category percentages over time.
            """
            return self.temporal_percentage_stacked_bar_plot(
                categorical_column, datetime_column, frequency, top_n
            )

        @tool
        def category_trend_plot(
            categorical_column: str,
            datetime_column: str,
            category: str,
            frequency: str = self.DEFAULT_FREQUENCY,
        ) -> plt.Figure:
            """
            Plot the temporal trend of one category.
            """
            return self.category_trend_plot(
                categorical_column, datetime_column, category, frequency
            )

        @tool
        def temporal_category_heatmap(
            categorical_column: str,
            datetime_column: str,
            frequency: str = self.DEFAULT_FREQUENCY,
            top_n: int = self.DEFAULT_TOP_N,
        ) -> plt.Figure:
            """
            Plot category frequencies across temporal periods.
            """
            return self.temporal_category_heatmap(
                categorical_column, datetime_column, frequency, top_n
            )

        @tool
        def temporal_percentage_heatmap(
            categorical_column: str,
            datetime_column: str,
            frequency: str = self.DEFAULT_FREQUENCY,
            top_n: int = self.DEFAULT_TOP_N,
        ) -> plt.Figure:
            """
            Plot category percentages across temporal periods.
            """
            return self.temporal_percentage_heatmap(
                categorical_column, datetime_column, frequency, top_n
            )

        @tool
        def category_presence_plot(
            categorical_column: str,
            datetime_column: str,
            frequency: str = self.DEFAULT_FREQUENCY,
            top_n: int = self.DEFAULT_TOP_N,
        ) -> plt.Figure:
            """
            Visualize category presence across temporal periods.
            """
            return self.category_presence_plot(
                categorical_column, datetime_column, frequency, top_n
            )

        @tool
        def temporal_missingness_plot(
            categorical_column: str,
            datetime_column: str,
            frequency: str = self.DEFAULT_FREQUENCY,
        ) -> plt.Figure:
            """
            Plot categorical missingness over time.
            """
            return self.temporal_missingness_plot(
                categorical_column, datetime_column, frequency
            )

        tools: list[Any] = [
            temporal_category_count_plot,
            temporal_percentage_plot,
            temporal_stacked_bar_plot,
            temporal_percentage_stacked_bar_plot,
            category_trend_plot,
            temporal_category_heatmap,
            temporal_percentage_heatmap,
            category_presence_plot,
            temporal_missingness_plot,
        ]
        self.logger.info(
            "Built %d categorical-datetime bivariate visualization tools.",
            len(tools),
        )
        return tools

# ====================================================
# Factory
# ====================================================

def build_categorical_datetime_bivariate_statistics_tools(df: pd.DataFrame,) -> list[Any]:
    """
    Create all categorical-datetime bivariate statistical LangChain
    tools for a DataFrame.

    Parameters
    ----------
    df:
        DataFrame that the tools will analyze.

    Returns
    -------
    list[Any]
        LangChain tools covering categorical-datetime bivariate analysis.

    Raises
    ------
    CustomException
        If the DataFrame is invalid.
    """
    engine = CategoricalDatetimeBivariateStatistics(df)
    engine.logger.info("Building categorical-datetime bivariate statistics tool collection.")
    tools = engine.build_tools()
    engine.logger.info("Built %d categorical-datetime bivariate statistics tools.",len(tools),)
    return tools


def build_categorical_datetime_bivariate_visualization_tools(df: pd.DataFrame,) -> list[Any]:
    """
    Build categorical-datetime bivariate visualization tools.

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
    engine = CategoricalDatetimeBivariateVisualization(df)
    engine.logger.info("Building categorical-datetime bivariate visualization tool collection.")
    tools = engine.build_tools()
    engine.logger.info("Built %d categorical-datetime bivariate visualization tools.",len(tools),)
    return tools


# ====================================================
# Build Tools
# ====================================================

def build_categorical_datetime_bivariate_tools(df: pd.DataFrame,) -> list[Any]:
    """
    Build all categorical-datetime bivariate tools.

    Includes
    --------
    - Categorical-datetime bivariate statistics
    - Categorical-datetime bivariate visualizations

    Parameters
    ----------
    df:
        Source DataFrame.

    Returns
    -------
    list[Any]
        Complete categorical-datetime bivariate tool collection.
    """
    tools: list[Any] = []
    tools.extend(build_categorical_datetime_bivariate_statistics_tools(df))
    tools.extend(build_categorical_datetime_bivariate_visualization_tools(df))
    return tools