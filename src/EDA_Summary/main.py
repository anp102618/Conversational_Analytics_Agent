from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import Any

from src.config import BASE_DIR
from src.Utils.columns_classifier import classify_columns
from src.Utils.exception_handler import CustomException
from src.Utils.logger_setup import get_log, track_performance

from src.Utils.columns_classifier import classify_columns


# ----------------------------------------------------------------------
# UNIVARIATE STATISTICS & SUMMARIZATION
# ----------------------------------------------------------------------

from src.EDA_Summary.Univariate_Analysis.Statistics.numeric import uni_num_stats
from src.EDA_Summary.Univariate_Analysis.Statistics.categorical import uni_cat_stats
from src.EDA_Summary.Univariate_Analysis.LLm_Summarisation.num_stat_summarization import (
    execute_univariate_num_eda,
)
from src.EDA_Summary.Univariate_Analysis.LLm_Summarisation.cat_stat_summarization import (
    execute_univariate_cat_eda,
)
from src.Utils.report_aggregation import eda_text_report_generation
# ----------------------------------------------------------------------
# BIVARIATE STATISTICS & SUMMARIZATION
# ----------------------------------------------------------------------

from src.EDA_Summary.Bivariate_Analysis.Statistics.num_num import num_num_stats
from src.EDA_Summary.Bivariate_Analysis.Statistics.num_cat import num_cat_stats
from src.EDA_Summary.Bivariate_Analysis.Statistics.cat_cat import cat_cat_stats
from src.EDA_Summary.Bivariate_Analysis.Statistics.num_dt import num_datetime_stats
from src.EDA_Summary.Bivariate_Analysis.Statistics.cat_dt import cat_datetime_stats

from src.EDA_Summary.Bivariate_Analysis.LLm_Summarisation.num_num_stat_summarization import (
    execute_bivariate_num_num_eda,
)
from src.EDA_Summary.Bivariate_Analysis.LLm_Summarisation.num_cat_stat_summarization import (
    execute_bivariate_num_cat_eda,
)
from src.EDA_Summary.Bivariate_Analysis.LLm_Summarisation.cat_cat_stat_summarization import (
    execute_bivariate_cat_cat_eda,
)
from src.EDA_Summary.Bivariate_Analysis.LLm_Summarisation.num_dt_stat_summarization import (
    execute_bivariate_num_datetime_eda,
)
from src.EDA_Summary.Bivariate_Analysis.LLm_Summarisation.cat_dt_stat_summarization import (
    execute_bivariate_cat_datetime_eda,
)

# ----------------------------------------------------------------------
# MULTIVARIATE STATISTICS & SUMMARIZATION
# ----------------------------------------------------------------------

from src.EDA_Summary.Multivariate_Analysis.Statistics.multivariate import (
    multivariate_stats,
)
from src.EDA_Summary.Multivariate_Analysis.LLm_Summarisation.multivariate_stat_summarization import (
    execute_multivariate_eda,
)

# ======================================================================
# LOGGER
# ======================================================================

LOG_DIR = BASE_DIR / "src" / "EDA_Summary" /  "logs"
logger = get_log("EDAStrategy", log_dir=LOG_DIR)


# ======================================================================
# EDA TYPE
# ======================================================================


class EDAType(str, Enum):
    """Supported EDA strategy types."""

    UNIVARIATE_NUMERIC = "univariate_numeric"
    UNIVARIATE_CATEGORICAL = "univariate_categorical"

    BIVARIATE_NUMERIC_NUMERIC = "bivariate_numeric_numeric"
    BIVARIATE_NUMERIC_CATEGORICAL = "bivariate_numeric_categorical"
    BIVARIATE_CATEGORICAL_CATEGORICAL = "bivariate_categorical_categorical"
    BIVARIATE_NUMERIC_DATETIME = "bivariate_numeric_datetime"
    BIVARIATE_CATEGORICAL_DATETIME = "bivariate_categorical_datetime"

    MULTIVARIATE = "multivariate"


# ======================================================================
# BASE STRATEGY
# ======================================================================


class EDAStrategy(ABC):
    """Abstract base class for all EDA strategies."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the strategy name."""
        raise NotImplementedError

    @abstractmethod
    def execute_statistics(self) -> Any:
        """Execute deterministic EDA statistics."""
        raise NotImplementedError

    @abstractmethod
    def execute_summary(self) -> Any:
        """Execute LLM summarization."""
        raise NotImplementedError

    @track_performance
    def execute(self) -> dict[str, Any]:
        """Execute statistics followed by LLM summarization."""
        try:
            logger.info("Starting EDA strategy: %s", self.name)
            statistics_result = self.execute_statistics()
            logger.info("Statistics completed: %s", self.name)

            summary_result = self.execute_summary()
            logger.info("LLM summarization completed: %s", self.name)

            return {
                "strategy": self.name,
                "statistics": statistics_result,
                "summary": summary_result,
                "status": "success",
            }

        except Exception as exc:
            logger.exception("EDA strategy failed: %s", self.name)
            raise CustomException(f"EDA strategy '{self.name}' failed: {exc}") from exc


# ======================================================================
# UNIVARIATE STRATEGIES
# ======================================================================


class NumericUnivariateStrategy(EDAStrategy):
    """Strategy for numerical univariate EDA."""

    @property
    def name(self) -> str:
        return EDAType.UNIVARIATE_NUMERIC.value

    def execute_statistics(self) -> Any:
        classify_columns()
        return uni_num_stats()

    def execute_summary(self) -> Any:
        return execute_univariate_num_eda()


class CategoricalUnivariateStrategy(EDAStrategy):
    """Strategy for categorical univariate EDA."""

    @property
    def name(self) -> str:
        return EDAType.UNIVARIATE_CATEGORICAL.value

    def execute_statistics(self) -> Any:
        return uni_cat_stats()

    def execute_summary(self) -> Any:
        return execute_univariate_cat_eda()


# ======================================================================
# BIVARIATE STRATEGIES
# ======================================================================


class NumericNumericStrategy(EDAStrategy):
    """Strategy for numeric-numeric EDA."""

    @property
    def name(self) -> str:
        return EDAType.BIVARIATE_NUMERIC_NUMERIC.value

    def execute_statistics(self) -> Any:
        return num_num_stats()

    def execute_summary(self) -> Any:
        return execute_bivariate_num_num_eda()


class NumericCategoricalStrategy(EDAStrategy):
    """Strategy for numeric-categorical EDA."""

    @property
    def name(self) -> str:
        return EDAType.BIVARIATE_NUMERIC_CATEGORICAL.value

    def execute_statistics(self) -> Any:
        return num_cat_stats()

    def execute_summary(self) -> Any:
        return execute_bivariate_num_cat_eda()


class CategoricalCategoricalStrategy(EDAStrategy):
    """Strategy for categorical-categorical EDA."""

    @property
    def name(self) -> str:
        return EDAType.BIVARIATE_CATEGORICAL_CATEGORICAL.value

    def execute_statistics(self) -> Any:
        return cat_cat_stats()

    def execute_summary(self) -> Any:
        return execute_bivariate_cat_cat_eda()


class NumericDatetimeStrategy(EDAStrategy):
    """Strategy for numeric-datetime EDA."""

    @property
    def name(self) -> str:
        return EDAType.BIVARIATE_NUMERIC_DATETIME.value

    def execute_statistics(self) -> Any:
        return num_datetime_stats()

    def execute_summary(self) -> Any:
        return execute_bivariate_num_datetime_eda()


class CategoricalDatetimeStrategy(EDAStrategy):
    """Strategy for categorical-datetime EDA."""

    @property
    def name(self) -> str:
        return EDAType.BIVARIATE_CATEGORICAL_DATETIME.value

    def execute_statistics(self) -> Any:
        return cat_datetime_stats()

    def execute_summary(self) -> Any:
        return execute_bivariate_cat_datetime_eda()


# ======================================================================
# MULTIVARIATE STRATEGY
# ======================================================================


class MultivariateStrategy(EDAStrategy):
    """Strategy for complete multivariate EDA."""

    def __init__(
        self,
        target: str | None = None,
        file_path: Path | str | None = None,
    ) -> None:
        self.target = target
        self.file_path = file_path

    @property
    def name(self) -> str:
        return EDAType.MULTIVARIATE.value

    def execute_statistics(self) -> Any:
        return multivariate_stats(file_path=self.file_path, target=self.target)

    def execute_summary(self) -> Any:
        return execute_multivariate_eda()


# ======================================================================
# STRATEGY FACTORY
# ======================================================================


class EDAStrategyFactory:
    """Factory responsible for creating EDA strategies."""

    @staticmethod
    def create(
        eda_type: EDAType,
        target: str | None = None,
        file_path: Path | str | None = None,
    ) -> EDAStrategy:
        if eda_type == EDAType.UNIVARIATE_NUMERIC:
            return NumericUnivariateStrategy()
        if eda_type == EDAType.UNIVARIATE_CATEGORICAL:
            return CategoricalUnivariateStrategy()
        if eda_type == EDAType.BIVARIATE_NUMERIC_NUMERIC:
            return NumericNumericStrategy()
        if eda_type == EDAType.BIVARIATE_NUMERIC_CATEGORICAL:
            return NumericCategoricalStrategy()
        if eda_type == EDAType.BIVARIATE_CATEGORICAL_CATEGORICAL:
            return CategoricalCategoricalStrategy()
        if eda_type == EDAType.BIVARIATE_NUMERIC_DATETIME:
            return NumericDatetimeStrategy()
        if eda_type == EDAType.BIVARIATE_CATEGORICAL_DATETIME:
            return CategoricalDatetimeStrategy()
        if eda_type == EDAType.MULTIVARIATE:
            return MultivariateStrategy(target=target, file_path=file_path)

        raise ValueError(f"Unsupported EDA strategy: {eda_type}")


# ======================================================================
# EDA CONTEXT
# ======================================================================


class EDAContext:
    """Context that executes the selected EDA strategy."""

    def __init__(self, strategy: EDAStrategy) -> None:
        self.strategy = strategy

    def set_strategy(self, strategy: EDAStrategy) -> None:
        self.strategy = strategy

    def execute(self) -> dict[str, Any]:
        return self.strategy.execute()


# ======================================================================
# EDA PIPELINE
# ======================================================================


class EDAPipeline:
    """Central EDA orchestration pipeline."""

    def __init__(
        self,
        target: str | None = None,
        file_path: Path | str | None = None,
    ) -> None:
        self.target = target
        self.file_path = file_path

    @track_performance
    def classify(self) -> Any:
        try:
            logger.info("Starting column classification.")
            result = classify_columns()
            logger.info("Column classification completed.")
            return result
        except Exception as exc:
            logger.exception("Column classification failed.")
            raise CustomException(f"Column classification failed: {exc}") from exc

    @track_performance
    def execute_strategy(self, eda_type: EDAType) -> dict[str, Any]:
        try:
            logger.info("Creating EDA strategy: %s", eda_type.value)
            strategy = EDAStrategyFactory.create(
                eda_type=eda_type,
                target=self.target,
                file_path=self.file_path,
            )
            context = EDAContext(strategy)
            result = context.execute()
            logger.info("Strategy completed: %s", eda_type.value)
            return result
        except Exception as exc:
            logger.exception("Strategy execution failed: %s", eda_type.value)
            raise CustomException(f"Strategy '{eda_type.value}' failed: {exc}") from exc

    @track_performance
    def run(
        self,
        selected_analyses: list[EDAType] | None = None,
        classify: bool = True,
    ) -> dict[str, Any]:
        try:
            logger.info("Starting unified EDA pipeline.")
            if classify:
                self.classify()

            if selected_analyses is None:
                selected_analyses = list(EDAType)

            results: dict[str, Any] = {}
            for eda_type in selected_analyses:
                logger.info("Executing EDA strategy: %s", eda_type.value)
                results[eda_type.value] = self.execute_strategy(eda_type)

            logger.info("Unified EDA pipeline completed.")
            return results
        except Exception as exc:
            logger.exception("Unified EDA pipeline failed.")
            raise CustomException(f"Unified EDA pipeline failed: {exc}") from exc


# ======================================================================
# PUBLIC IMPLEMENTATION FUNCTIONS
# ======================================================================


@track_performance
def run_eda(
    target: str | None = None,
    file_path: Path | str | None = None,
    selected_analyses: list[EDAType] | None = None,
) -> dict[str, Any]:
    pipeline = EDAPipeline(target=target, file_path=file_path)
    return pipeline.run(selected_analyses=selected_analyses)


def run_univariate_eda() -> dict[str, Any]:
    return run_eda(
        selected_analyses=[
            EDAType.UNIVARIATE_NUMERIC,
            EDAType.UNIVARIATE_CATEGORICAL,
        ]
    )


def run_bivariate_eda() -> dict[str, Any]:
    return run_eda(
        selected_analyses=[
            EDAType.BIVARIATE_NUMERIC_NUMERIC,
            EDAType.BIVARIATE_NUMERIC_CATEGORICAL,
            EDAType.BIVARIATE_CATEGORICAL_CATEGORICAL,
            EDAType.BIVARIATE_NUMERIC_DATETIME,
            EDAType.BIVARIATE_CATEGORICAL_DATETIME,
        ]
    )


def run_multivariate_eda(
    target: str | None = None,
    file_path: Path | str | None = None,
) -> dict[str, Any]:
    return run_eda(
        target=target,
        file_path=file_path,
        selected_analyses=[EDAType.MULTIVARIATE],
    )


def run_all_eda(
    target: str | None = None,
    file_path: Path | str | None = None,
) -> dict[str, Any]:
    run_eda(target=target, file_path=file_path)
    return eda_text_report_generation()


# ======================================================================
# MAIN
# ======================================================================

if __name__ == "__main__":
    results = run_all_eda(target="Returned")

    print("\n==========================================")
    print("        COMPLETE EDA PIPELINE")
    print("==========================================")

    for strategy_name, result in results.items():
        print(f"\n{strategy_name}")
        print(f"Status: {result.get('status')}")