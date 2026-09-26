#====================================================
# Numerical-Datetime Bivariate Statistics
#====================================================

"""
Numeric-Datetime Bivariate Exploratory Data Analysis.

Deterministic statistical analysis between numeric and datetime features.

Analysis includes:
- Temporal aggregation
- Temporal trend statistics
- Spearman correlation with time
- Rolling statistics
- Period-based summaries
- Highest/lowest temporal periods
- Temporal change analysis
- Missing values by period

The generated results are deterministic and suitable for downstream
EDA, RAG, SQL, and Pandas Data Agent workflows.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from langchain_core.tools import tool
from scipy.stats import spearmanr
from src.Utils.exception_handler import CustomException
from src.Utils.logger_setup import get_log, track_performance


class NumericDatetimeBivariateStatistics:
    """
    Deterministic numeric-datetime bivariate statistical analyzer.

    Parameters
    ----------
    df:
        Source pandas DataFrame.

    Notes
    -----
    The DataFrame remains internal to the analyzer. LangChain tools
    expose only column names and analysis parameters.
    """

    DEFAULT_FREQUENCY = "M"
    DEFAULT_ROLLING_WINDOW = 7
    DEFAULT_MIN_OBSERVATIONS = 3
    DEFAULT_TOP_N = 5

    def __init__(self, df: pd.DataFrame) -> None:
        """
        Initialize numeric-datetime analyzer.

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

    def _validate_columns(
        self, numeric_column: str, datetime_column: str
    ) -> None:
        """
        Validate numeric and datetime columns.
        """
        if numeric_column not in self.df.columns:
            raise CustomException(f"Column '{numeric_column}' does not exist.")
        if datetime_column not in self.df.columns:
            raise CustomException(f"Column '{datetime_column}' does not exist.")
        if not pd.api.types.is_numeric_dtype(self.df[numeric_column]):
            raise CustomException(f"Column '{numeric_column}' must be numeric.")
        if not pd.api.types.is_datetime64_any_dtype(self.df[datetime_column]):
            raise CustomException(
                f"Column '{datetime_column}' must be datetime."
            )

    def _get_pair(
        self, numeric_column: str, datetime_column: str
    ) -> pd.DataFrame:
        """
        Return valid numeric-datetime observations.
        """
        self._validate_columns(numeric_column, datetime_column)
        data = self.df[[numeric_column, datetime_column]].dropna(
            subset=[numeric_column, datetime_column]
        )
        if data.empty:
            raise CustomException(
                "No valid observations remain after removing "
                "missing numeric or datetime values."
            )
        return data.sort_values(datetime_column)

    @track_performance
    def temporal_aggregation(
        self,
        numeric_column: str,
        datetime_column: str,
        frequency: str = DEFAULT_FREQUENCY,
    ) -> dict[str, Any]:
        """
        Aggregate a numeric feature over time.

        Supported pandas frequencies include:
        - D: Daily
        - W: Weekly
        - M: Monthly
        - Q: Quarterly
        - Y: Yearly
        """
        data = self._get_pair(numeric_column, datetime_column)
        if not frequency:
            raise CustomException("frequency cannot be empty.")
        try:
            data = data.set_index(datetime_column)
            grouped = data[numeric_column].resample(frequency)
            result = pd.DataFrame({
                "count": grouped.count(),
                "mean": grouped.mean(),
                "median": grouped.median(),
                "std": grouped.std(),
                "min": grouped.min(),
                "max": grouped.max(),
                "sum": grouped.sum(),
            }).reset_index()
        except Exception as exc:
            raise CustomException(
                f"Temporal aggregation failed for frequency '{frequency}': {exc}"
            ) from exc
        result = result.dropna(subset=["count"])
        return {
            "numeric_column": numeric_column,
            "datetime_column": datetime_column,
            "frequency": frequency,
            "observations": int(len(data)),
            "periods": result.to_dict(orient="records"),
        }

    @track_performance
    def temporal_trend_statistics(
        self, numeric_column: str, datetime_column: str
    ) -> dict[str, Any]:
        """
        Calculate descriptive statistics describing the temporal trend.

        The datetime column is converted to elapsed seconds from the
        first observation and Spearman correlation is calculated against
        the numeric feature.
        """
        data = self._get_pair(numeric_column, datetime_column)
        if len(data) < 2:
            raise CustomException(
                "At least two observations are required for "
                "temporal trend analysis."
            )
        elapsed_seconds = (
            data[datetime_column] - data[datetime_column].min()
        ).dt.total_seconds()
        correlation, p_value = spearmanr(elapsed_seconds, data[numeric_column])
        first_value = float(data[numeric_column].iloc[0])
        last_value = float(data[numeric_column].iloc[-1])
        absolute_change = last_value - first_value
        percentage_change = (
            absolute_change / abs(first_value) * 100 if first_value != 0 else None
        )
        return {
            "numeric_column": numeric_column,
            "datetime_column": datetime_column,
            "observations": int(len(data)),
            "start_date": str(data[datetime_column].min()),
            "end_date": str(data[datetime_column].max()),
            "first_value": first_value,
            "last_value": last_value,
            "absolute_change": float(absolute_change),
            "percentage_change": (
                float(percentage_change) if percentage_change is not None else None
            ),
            "spearman_correlation": (
                float(correlation) if not np.isnan(correlation) else None
            ),
            "spearman_p_value": (
                float(p_value) if not np.isnan(p_value) else None
            ),
        }

    @track_performance
    def spearman_correlation_with_time(
        self, numeric_column: str, datetime_column: str
    ) -> dict[str, Any]:
        """
        Calculate Spearman correlation between the numeric feature
        and elapsed time.
        """
        data = self._get_pair(numeric_column, datetime_column)
        if len(data) < 3:
            raise CustomException(
                "At least three observations are required for "
                "Spearman correlation."
            )
        elapsed_seconds = (
            data[datetime_column] - data[datetime_column].min()
        ).dt.total_seconds()
        correlation, p_value = spearmanr(elapsed_seconds, data[numeric_column])
        correlation_value = (
            None if np.isnan(correlation) else float(correlation)
        )
        p_value_value = None if np.isnan(p_value) else float(p_value)
        return {
            "numeric_column": numeric_column,
            "datetime_column": datetime_column,
            "method": "spearman",
            "correlation": correlation_value,
            "p_value": p_value_value,
            "observations": int(len(data)),
        }

    @track_performance
    def rolling_statistics(
        self,
        numeric_column: str,
        datetime_column: str,
        window: int = DEFAULT_ROLLING_WINDOW,
        min_observations: int = DEFAULT_MIN_OBSERVATIONS,
    ) -> dict[str, Any]:
        """
        Calculate rolling statistics over chronologically ordered data.

        Parameters
        ----------
        window:
            Number of observations in the rolling window.

        min_observations:
            Minimum observations required to calculate a rolling value.
        """
        if window <= 0:
            raise CustomException("window must be greater than zero.")
        if min_observations <= 0:
            raise CustomException("min_observations must be greater than zero.")
        if min_observations > window:
            raise CustomException(
                "min_observations cannot be greater than window."
            )
        data = self._get_pair(numeric_column, datetime_column)
        rolling = data[numeric_column].rolling(
            window=window, min_periods=min_observations
        )
        result = pd.DataFrame({
            datetime_column: data[datetime_column],
            numeric_column: data[numeric_column],
            "rolling_mean": rolling.mean(),
            "rolling_median": rolling.median(),
            "rolling_std": rolling.std(),
            "rolling_min": rolling.min(),
            "rolling_max": rolling.max(),
        }).dropna(subset=["rolling_mean"])
        return {
            "numeric_column": numeric_column,
            "datetime_column": datetime_column,
            "window": int(window),
            "min_observations": int(min_observations),
            "observations": int(len(result)),
            "rolling_statistics": result.to_dict(orient="records"),
        }

    @track_performance
    def period_analysis(
        self,
        numeric_column: str,
        datetime_column: str,
        frequency: str = DEFAULT_FREQUENCY,
        top_n: int = DEFAULT_TOP_N,
    ) -> dict[str, Any]:
        """
        Identify highest and lowest temporal periods based on
        aggregated mean values.
        """
        if top_n <= 0:
            raise CustomException("top_n must be greater than zero.")
        data = self._get_pair(numeric_column, datetime_column)
        try:
            data = data.set_index(datetime_column)
            aggregated = (
                data[numeric_column]
                .resample(frequency)
                .agg(["count", "mean", "median", "min", "max"])
                .dropna(subset=["mean"])
                .reset_index()
            )
        except Exception as exc:
            raise CustomException(f"Period analysis failed: {exc}") from exc
        if aggregated.empty:
            raise CustomException("No valid temporal periods were generated.")
        highest = aggregated.sort_values("mean", ascending=False).head(top_n)
        lowest = aggregated.sort_values("mean", ascending=True).head(top_n)
        return {
            "numeric_column": numeric_column,
            "datetime_column": datetime_column,
            "frequency": frequency,
            "top_n": int(top_n),
            "highest_periods": highest.to_dict(orient="records"),
            "lowest_periods": lowest.to_dict(orient="records"),
        }

    @track_performance
    def temporal_change_analysis(
        self,
        numeric_column: str,
        datetime_column: str,
        frequency: str = DEFAULT_FREQUENCY,
    ) -> dict[str, Any]:
        """
        Calculate period-over-period absolute and percentage changes.
        """
        data = self._get_pair(numeric_column, datetime_column)
        try:
            data = data.set_index(datetime_column)
            aggregated = (
                data[numeric_column]
                .resample(frequency)
                .mean()
                .dropna()
                .to_frame("value")
            )
        except Exception as exc:
            raise CustomException(
                f"Temporal change analysis failed: {exc}"
            ) from exc
        if len(aggregated) < 2:
            raise CustomException(
                "At least two temporal periods are required "
                "for change analysis."
            )
        aggregated["absolute_change"] = aggregated["value"].diff()
        previous = aggregated["value"].shift(1)
        aggregated["percentage_change"] = np.where(
            previous != 0,
            aggregated["absolute_change"] / previous.abs() * 100,
            np.nan,
        )
        aggregated = aggregated.reset_index()
        return {
            "numeric_column": numeric_column,
            "datetime_column": datetime_column,
            "frequency": frequency,
            "periods": aggregated.to_dict(orient="records"),
        }

    @track_performance
    def missing_values_by_period(
        self,
        numeric_column: str,
        datetime_column: str,
        frequency: str = DEFAULT_FREQUENCY,
    ) -> dict[str, Any]:
        """
        Analyze missing numeric observations across time periods.

        Unlike most methods, this method intentionally preserves missing
        numeric values so that missingness can be measured.
        """
        self._validate_columns(numeric_column, datetime_column)
        data = self.df[[numeric_column, datetime_column]].dropna(
            subset=[datetime_column]
        )
        if data.empty:
            raise CustomException("No valid datetime observations available.")
        data["is_missing"] = data[numeric_column].isna()
        try:
            grouped = (
                data.set_index(datetime_column)
                .resample(frequency)["is_missing"]
                .agg(["count", "sum", "mean"])
                .rename(columns={
                    "count": "total_observations",
                    "sum": "missing_count",
                    "mean": "missing_rate",
                })
                .reset_index()
            )
        except Exception as exc:
            raise CustomException(
                f"Missing-value temporal analysis failed: {exc}"
            ) from exc
        grouped["missing_rate"] = grouped["missing_rate"] * 100
        return {
            "numeric_column": numeric_column,
            "datetime_column": datetime_column,
            "frequency": frequency,
            "periods": grouped.to_dict(orient="records"),
        }

    def build_tools(self) -> list[Any]:
        """
        Build LangChain tools for numeric-datetime bivariate
        statistical analysis.

        Returns
        -------
        list[Any]
            List of LangChain tools.
        """
        @tool
        def temporal_aggregation(
            numeric_column: str,
            datetime_column: str,
            frequency: str = "M",
        ) -> dict[str, Any]:
            """
            Aggregate a numeric column over time using a specified
            temporal frequency.
            """
            return self.temporal_aggregation(
                numeric_column, datetime_column, frequency
            )

        @tool
        def temporal_trend_statistics(
            numeric_column: str, datetime_column: str
        ) -> dict[str, Any]:
            """
            Calculate temporal trend statistics for a numeric column.
            """
            return self.temporal_trend_statistics(
                numeric_column, datetime_column
            )

        @tool
        def spearman_correlation_with_time(
            numeric_column: str, datetime_column: str
        ) -> dict[str, Any]:
            """
            Calculate Spearman correlation between a numeric feature
            and elapsed time.
            """
            return self.spearman_correlation_with_time(
                numeric_column, datetime_column
            )

        @tool
        def rolling_statistics(
            numeric_column: str,
            datetime_column: str,
            window: int = 7,
            min_observations: int = 3,
        ) -> dict[str, Any]:
            """
            Calculate rolling mean, median, standard deviation,
            minimum, and maximum over chronological observations.
            """
            return self.rolling_statistics(
                numeric_column, datetime_column, window, min_observations
            )

        @tool
        def period_analysis(
            numeric_column: str,
            datetime_column: str,
            frequency: str = "M",
            top_n: int = 5,
        ) -> dict[str, Any]:
            """
            Identify highest and lowest temporal periods based on
            aggregated numeric values.
            """
            return self.period_analysis(
                numeric_column, datetime_column, frequency, top_n
            )

        @tool
        def temporal_change_analysis(
            numeric_column: str,
            datetime_column: str,
            frequency: str = "M",
        ) -> dict[str, Any]:
            """
            Calculate period-over-period absolute and percentage
            changes for a numeric feature.
            """
            return self.temporal_change_analysis(
                numeric_column, datetime_column, frequency
            )

        @tool
        def missing_values_by_period(
            numeric_column: str,
            datetime_column: str,
            frequency: str = "M",
        ) -> dict[str, Any]:
            """
            Analyze numeric missing values across temporal periods.
            """
            return self.missing_values_by_period(
                numeric_column, datetime_column, frequency
            )

        tools = [
            temporal_aggregation,
            temporal_trend_statistics,
            spearman_correlation_with_time,
            rolling_statistics,
            period_analysis,
            temporal_change_analysis,
            missing_values_by_period,
        ]
        self.logger.info(
            "Built %d numeric-datetime bivariate statistical tools.", len(tools)
        )
        return tools


#===========================================================================
#====================================================
# Numerical-Datetime Bivariate Visualization
#====================================================

"""
Numeric-Datetime Bivariate Visualization.
Deterministic visualization between numeric and datetime features
using Seaborn and Matplotlib.
Visualizations:
- Time series plot
- Aggregated temporal trend plot
- Rolling statistics plot
- Period distribution plot
- Temporal box plot
- Temporal missingness plot
"""
from __future__ import annotations
from typing import Any
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from langchain_core.tools import tool
from src.Utils.exception_handler import CustomException
from src.Utils.logger_setup import get_log, track_performance

class NumericDatetimeBivariateVisualization:
    """
    Deterministic numeric-datetime bivariate visualization engine.
    Parameters
    ----------
    df:
        Source pandas DataFrame.
    Notes
    -----
    The DataFrame remains internal to the visualization engine.
    LangChain tools expose only column names and visualization
    parameters.
    """
    DEFAULT_FREQUENCY = "M"
    DEFAULT_ROLLING_WINDOW = 7
    DEFAULT_MIN_OBSERVATIONS = 3
    DEFAULT_TOP_N = 5
    DEFAULT_FIGSIZE = (12, 6)

    def __init__(self, df: pd.DataFrame) -> None:
        """
        Initialize numeric-datetime visualization engine.
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

    def _validate_columns(
        self, numeric_column: str, datetime_column: str
    ) -> None:
        """
        Validate numeric and datetime columns.
        """
        if numeric_column not in self.df.columns:
            raise CustomException(f"Column '{numeric_column}' does not exist.")
        if datetime_column not in self.df.columns:
            raise CustomException(f"Column '{datetime_column}' does not exist.")
        if not pd.api.types.is_numeric_dtype(self.df[numeric_column]):
            raise CustomException(f"Column '{numeric_column}' must be numeric.")
        if not pd.api.types.is_datetime64_any_dtype(self.df[datetime_column]):
            raise CustomException(
                f"Column '{datetime_column}' must be datetime."
            )

    def _get_pair(
        self, numeric_column: str, datetime_column: str
    ) -> pd.DataFrame:
        """
        Return valid numeric-datetime observations sorted by time.
        """
        self._validate_columns(numeric_column, datetime_column)
        data = self.df[[numeric_column, datetime_column]].dropna(
            subset=[numeric_column, datetime_column]
        )
        if data.empty:
            raise CustomException(
                "No valid observations remain after removing "
                "missing numeric or datetime values."
            )
        return data.sort_values(datetime_column)

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
        plt.xticks(rotation=45)
        plt.tight_layout()

    @track_performance
    def time_series_plot(
        self,
        numeric_column: str,
        datetime_column: str,
        figsize: tuple[int, int] | None = None,
    ) -> plt.Figure:
        """
        Plot the numeric feature against the datetime feature.
        Each valid observation is represented chronologically.
        """
        data = self._get_pair(numeric_column, datetime_column)
        fig, ax = self._create_figure(figsize)
        sns.lineplot(data=data, x=datetime_column, y=numeric_column, ax=ax)
        self._format_axis(
            ax, f"{numeric_column} Over Time", datetime_column, numeric_column
        )
        return fig

    @track_performance
    def temporal_trend_plot(
        self,
        numeric_column: str,
        datetime_column: str,
        frequency: str = DEFAULT_FREQUENCY,
        figsize: tuple[int, int] | None = None,
    ) -> plt.Figure:
        """
        Plot the mean numeric value aggregated over time.
        """
        data = self._get_pair(numeric_column, datetime_column)
        try:
            data = data.set_index(datetime_column)
            aggregated = (
                data[numeric_column]
                .resample(frequency)
                .mean()
                .dropna()
                .reset_index()
            )
        except Exception as exc:
            raise CustomException(
                f"Temporal trend aggregation failed: {exc}"
            ) from exc
        if aggregated.empty:
            raise CustomException(
                "No temporal observations available for plotting."
            )
        fig, ax = self._create_figure(figsize)
        sns.lineplot(
            data=aggregated,
            x=datetime_column,
            y=numeric_column,
            marker="o",
            ax=ax,
        )
        self._format_axis(
            ax,
            f"{numeric_column} Temporal Trend ({frequency})",
            datetime_column,
            f"Mean {numeric_column}",
        )
        return fig

    @track_performance
    def rolling_trend_plot(
        self,
        numeric_column: str,
        datetime_column: str,
        window: int = DEFAULT_ROLLING_WINDOW,
        min_observations: int = DEFAULT_MIN_OBSERVATIONS,
        figsize: tuple[int, int] | None = None,
    ) -> plt.Figure:
        """
        Plot the original numeric values together with a rolling mean.
        """
        if window <= 0:
            raise CustomException("window must be greater than zero.")
        if min_observations <= 0:
            raise CustomException("min_observations must be greater than zero.")
        if min_observations > window:
            raise CustomException(
                "min_observations cannot be greater than window."
            )
        data = self._get_pair(numeric_column, datetime_column)
        data["rolling_mean"] = (
            data[numeric_column]
            .rolling(window=window, min_periods=min_observations)
            .mean()
        )
        fig, ax = self._create_figure(figsize)
        sns.lineplot(
            data=data,
            x=datetime_column,
            y=numeric_column,
            alpha=0.4,
            ax=ax,
            label="Observed",
        )
        sns.lineplot(
            data=data,
            x=datetime_column,
            y="rolling_mean",
            ax=ax,
            label=f"Rolling Mean ({window})",
        )
        self._format_axis(
            ax, f"{numeric_column} Rolling Trend", datetime_column, numeric_column
        )
        ax.legend()
        return fig

    @track_performance
    def temporal_distribution_plot(
        self,
        numeric_column: str,
        datetime_column: str,
        frequency: str = DEFAULT_FREQUENCY,
        figsize: tuple[int, int] | None = None,
    ) -> plt.Figure:
        """
        Plot the distribution of the numeric feature across
        temporal periods.
        """
        data = self._get_pair(numeric_column, datetime_column)
        data["period"] = data[datetime_column].dt.to_period(frequency).astype(str)
        fig, ax = self._create_figure(figsize)
        sns.boxplot(data=data, x="period", y=numeric_column, ax=ax)
        self._format_axis(
            ax,
            f"{numeric_column} Distribution by Period",
            f"Period ({frequency})",
            numeric_column,
        )
        return fig

    @track_performance
    def temporal_box_plot(
        self,
        numeric_column: str,
        datetime_column: str,
        frequency: str = DEFAULT_FREQUENCY,
        top_n: int = DEFAULT_TOP_N,
        figsize: tuple[int, int] | None = None,
    ) -> plt.Figure:
        """
        Create a box plot of numeric values across the most recent
        temporal periods.
        Parameters
        ----------
        top_n:
            Number of most recent periods to display.
        """
        if top_n <= 0:
            raise CustomException("top_n must be greater than zero.")
        data = self._get_pair(numeric_column, datetime_column)
        data["period"] = data[datetime_column].dt.to_period(frequency)
        periods = data["period"].drop_duplicates().sort_values().tail(top_n)
        data = data[data["period"].isin(periods)].copy()
        data["period"] = data["period"].astype(str)
        if data.empty:
            raise CustomException("No data available for temporal box plot.")
        fig, ax = self._create_figure(figsize)
        sns.boxplot(data=data, x="period", y=numeric_column, ax=ax)
        self._format_axis(
            ax,
            f"{numeric_column} Distribution Across Recent Periods",
            f"Period ({frequency})",
            numeric_column,
        )
        return fig

    @track_performance
    def temporal_missingness_plot(
        self,
        numeric_column: str,
        datetime_column: str,
        frequency: str = DEFAULT_FREQUENCY,
        figsize: tuple[int, int] | None = None,
    ) -> plt.Figure:
        """
        Plot the percentage of missing numeric observations
        across temporal periods.
        Missing numeric values are intentionally preserved.
        """
        self._validate_columns(numeric_column, datetime_column)
        data = (
            self.df[[numeric_column, datetime_column]]
            .dropna(subset=[datetime_column])
            .copy()
        )
        if data.empty:
            raise CustomException("No valid datetime observations available.")
        data["is_missing"] = data[numeric_column].isna()
        try:
            missing_rate = (
                data.set_index(datetime_column)
                .resample(frequency)["is_missing"]
                .mean()
                .mul(100)
                .dropna()
                .reset_index(name="missing_rate")
            )
        except Exception as exc:
            raise CustomException(
                f"Temporal missingness aggregation failed: {exc}"
            ) from exc
        if missing_rate.empty:
            raise CustomException(
                "No temporal missingness observations available."
            )
        fig, ax = self._create_figure(figsize)
        sns.lineplot(
            data=missing_rate,
            x=datetime_column,
            y="missing_rate",
            marker="o",
            ax=ax,
        )
        self._format_axis(
            ax,
            f"Missingness of {numeric_column} Over Time",
            datetime_column,
            "Missing Rate (%)",
        )
        return fig

    def build_tools(self) -> list[Any]:
        """
        Build LangChain tools for numeric-datetime bivariate
        visualization.
        Returns
        -------
        list[Any]
            List of LangChain visualization tools.
        """
        @tool
        def time_series_plot(
            numeric_column: str, datetime_column: str
        ) -> plt.Figure:
            """
            Plot a numeric column against a datetime column.
            """
            return self.time_series_plot(numeric_column, datetime_column)

        @tool
        def temporal_trend_plot(
            numeric_column: str,
            datetime_column: str,
            frequency: str = "M",
        ) -> plt.Figure:
            """
            Plot the mean numeric value aggregated over time.
            """
            return self.temporal_trend_plot(
                numeric_column, datetime_column, frequency
            )

        @tool
        def rolling_trend_plot(
            numeric_column: str,
            datetime_column: str,
            window: int = 7,
            min_observations: int = 3,
        ) -> plt.Figure:
            """
            Plot observed numeric values with a rolling mean.
            """
            return self.rolling_trend_plot(
                numeric_column, datetime_column, window, min_observations
            )

        @tool
        def temporal_distribution_plot(
            numeric_column: str,
            datetime_column: str,
            frequency: str = "M",
        ) -> plt.Figure:
            """
            Plot numeric distributions across temporal periods.
            """
            return self.temporal_distribution_plot(
                numeric_column, datetime_column, frequency
            )

        @tool
        def temporal_box_plot(
            numeric_column: str,
            datetime_column: str,
            frequency: str = "M",
            top_n: int = 5,
        ) -> plt.Figure:
            """
            Create box plots of numeric values across recent
            temporal periods.
            """
            return self.temporal_box_plot(
                numeric_column, datetime_column, frequency, top_n
            )

        @tool
        def temporal_missingness_plot(
            numeric_column: str,
            datetime_column: str,
            frequency: str = "M",
        ) -> plt.Figure:
            """
            Plot the percentage of missing numeric observations
            across temporal periods.
            """
            return self.temporal_missingness_plot(
                numeric_column, datetime_column, frequency
            )

        tools = [
            time_series_plot,
            temporal_trend_plot,
            rolling_trend_plot,
            temporal_distribution_plot,
            temporal_box_plot,
            temporal_missingness_plot,
        ]
        self.logger.info(
            "Built %d numeric-datetime bivariate visualization tools.", len(tools)
        )
        return tools

    
# ====================================================
# Factory
# ====================================================

def build_numeric_datetime_bivariate_statistics_tools(df: pd.DataFrame,) -> list[Any]:
    """
    Create all numeric-datetime bivariate statistical LangChain
    tools for a DataFrame.
    Parameters
    ----------
    df:
        DataFrame that the tools will analyze.
    Returns
    -------
    list[Any]
        LangChain tools covering numeric-datetime bivariate analysis.
    Raises
    ------
    CustomException
        If the DataFrame is invalid.
    """
    engine = NumericDatetimeBivariateStatistics(df)
    engine.logger.info("Building numeric-datetime bivariate statistics tool collection.")
    tools = engine.build_tools()
    engine.logger.info("Built %d numeric-datetime bivariate statistics tools.", len(tools))
    return tools

def build_numeric_datetime_bivariate_visualization_tools(df: pd.DataFrame,) -> list[Any]:
    """
    Build numeric-datetime bivariate visualization tools.
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
    engine = NumericDatetimeBivariateVisualization(df)
    engine.logger.info("Building numeric-datetime bivariate visualization tool collection.")
    tools = engine.build_tools()
    engine.logger.info("Built %d numeric-datetime bivariate visualization tools.", len(tools))
    return tools

# ====================================================
# Build Tools
# ====================================================
def build_numeric_datetime_bivariate_tools(df: pd.DataFrame) -> list[Any]:
    """
    Build all numeric-datetime bivariate tools.
    Includes
    --------
    - Numeric-datetime bivariate statistics
    - Numeric-datetime bivariate visualizations
    Parameters
    ----------
    df:
        Source DataFrame.
    Returns
    -------
    list[Any]
        Complete numeric-datetime bivariate tool collection.
    """
    tools: list[Any] = []
    tools.extend(build_numeric_datetime_bivariate_statistics_tools(df))
    tools.extend(build_numeric_datetime_bivariate_visualization_tools(df))
    return tools
