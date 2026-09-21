# src/EDA_Summary/Bivariate_Analysis/summarize_numeric_numeric.py
from pathlib import Path

import pandas as pd

from src.config import BASE_DIR
from src.Utils.exception_handler import CustomException
from src.Utils.llm_variants import generate_response
from src.Utils.logger_setup import get_log, track_performance
from src.EDA_Summary.Prompts.bivariate_prompts import (
    get_numeric_numeric_prompt,
    SYSTEM_INSTRUCTION_NUM_NUM,
)

LOG_DIR = BASE_DIR / "src" / "EDA_Summary" / "Bivariate_Analysis" / "logs"
WORKBOOK_PATH = (
    BASE_DIR / "src" / "EDA_Summary" / "Bivariate_Analysis"
    / "Reports" / "excel_reports" / "num_num_eda.xlsx"
)
OUTPUT_TEXT_PATH = (
    BASE_DIR / "src" / "EDA_Summary" / "Bivariate_Analysis"
    / "Reports" / "text_reports" / "num_num_eda.txt"
)

logger = get_log("NumericNumericSummarizer", log_dir=LOG_DIR)

@track_performance
def build_workbook_context(excel_data: dict[str, pd.DataFrame]) -> str:
    """Convert all Excel worksheets into markdown-formatted LLM context."""
    parts: list[str] = []
    for sheet_name, df in excel_data.items():
        parts.append(
            f"\n\n==============================\n"
            f"WORKSHEET / METRIC TABLE: {sheet_name}\n"
            f"==============================\n"
        )
        parts.append(
            "No data available in this worksheet.\n"
            if df.empty
            else df.to_markdown(index=False)
        )
    return "".join(parts)


@track_performance
def summarize_numeric_numeric_workbook(file_path: Path | str) -> str:
    """
    Read pre-calculated numeric-numeric EDA metrics from Excel
    and generate an LLM-based executive summary.
    """
    file_path = Path(file_path)

    if not file_path.exists():
        logger.error("File not found: %s", file_path)
        raise CustomException(FileNotFoundError(f"File not found: {file_path}"), logger)

    try:
        logger.info("Reading numeric-numeric EDA workbook from %s", file_path)
        excel_data = pd.read_excel(file_path, sheet_name=None)
    except Exception as exc:
        logger.error("Error reading Excel workbook: %s", exc)
        raise CustomException(exc, logger)

    if not excel_data:
        logger.error("Excel workbook contains no worksheets.")
        raise CustomException(ValueError("Excel workbook contains no worksheets."), logger)

    try:
        workbook_text_context = build_workbook_context(excel_data)
    except Exception as exc:
        logger.error("Failed to build workbook text context: %s", exc)
        raise CustomException(exc, logger)

    user_prompt = get_numeric_numeric_prompt(workbook_text_context)

    logger.info("Sending numeric-numeric metrics to resilient LLM pipeline...")
    try:
        summary = generate_response(
            system_prompt=SYSTEM_INSTRUCTION_NUM_NUM,
            user_prompt=user_prompt,
        )
    except Exception as exc:
        logger.error("LLM summarization failed: %s", exc)
        raise CustomException(exc, logger)

    if not summary:
        logger.error("LLM returned an empty summary.")
        raise CustomException(
            ValueError("LLM returned an empty numeric-numeric summary."), logger
        )

    try:
        OUTPUT_TEXT_PATH.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Saving numeric-numeric summary to %s", OUTPUT_TEXT_PATH)
        OUTPUT_TEXT_PATH.write_text(summary, encoding="utf-8")
    except Exception as exc:
        logger.error("Failed to save summary: %s", exc)
        raise CustomException(exc, logger)

    logger.info("Numeric-numeric EDA summary generated successfully.")
    return summary


@track_performance
def execute_bivariate_num_num_eda() -> None:
    """Execute numeric-numeric EDA summarization."""
    try:
        summary_result = summarize_numeric_numeric_workbook(WORKBOOK_PATH)
        if summary_result:
            print("\n=== LLM NUMERIC-NUMERIC EDA SUMMARY ===\n")
            print(summary_result)
    except Exception as exc:
        logger.exception("Execution failed: %s", exc)


if __name__ == "__main__":
    execute_bivariate_num_num_eda()