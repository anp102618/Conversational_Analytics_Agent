# src/EDA_Summary/Bivariate_Analysis/summarize_numeric_datetime.py
from pathlib import Path
import pandas as pd
from src.config import BASE_DIR
from src.Utils.exception_handler import CustomException
from src.Utils.llm_variants import generate_response
from src.Utils.logger_setup import get_log, track_performance
from src.EDA_Summary.Prompts.bivariate_prompts import (
    SYSTEM_INSTRUCTION_NUM_DATETIME,
    get_numeric_datetime_prompt,
)

LOG_DIR = BASE_DIR / "src/EDA_Summary/Bivariate_Analysis/logs"
WORKBOOK_PATH = (
    BASE_DIR / "src/EDA_Summary/Bivariate_Analysis/Reports/excel_reports/num_datetime_eda.xlsx"
)
OUTPUT_TEXT_PATH = (
    BASE_DIR / "src/EDA_Summary/Bivariate_Analysis/Reports/text_reports/num_datetime_eda.txt"
)

logger = get_log("NumericDatetimeSummarizer", log_dir=LOG_DIR)


def build_workbook_context(excel_data: dict[str, pd.DataFrame]) -> str:
    """Convert all temporal EDA worksheets into markdown context for the LLM."""
    parts = []
    for sheet_name, df in excel_data.items():
        parts.append(
            f"\n\n==============================\n"
            f"TEMPORAL METRIC: {sheet_name.upper()}\n"
            f"==============================\n"
        )
        parts.append(
            "No data available in this metric section.\n"
            if df.empty
            else df.to_markdown(index=False)
        )
    return "".join(parts)


@track_performance
def summarize_num_datetime_workbook(file_path: Path | str) -> str:
    """
    Read the pre-calculated numerical-datetime EDA workbook and generate
    an LLM-based temporal analysis summary.
    """
    file_path = Path(file_path)

    if not file_path.exists():
        logger.error("NUM-DATETIME workbook not found: %s", file_path)
        raise CustomException(FileNotFoundError(f"File not found: {file_path}"), logger)
    if not file_path.is_file():
        logger.error("NUM-DATETIME workbook path is not a file: %s", file_path)
        raise CustomException(ValueError(f"Invalid workbook path: {file_path}"), logger)

    try:
        logger.info("Reading NUM-DATETIME EDA workbook from %s", file_path)
        excel_data = pd.read_excel(file_path, sheet_name=None)
    except Exception as exc:
        logger.error("Error reading NUM-DATETIME Excel workbook: %s", exc)
        raise CustomException(exc, logger)

    if not excel_data:
        logger.error("NUM-DATETIME workbook contains no worksheets.")
        raise CustomException(ValueError("NUM-DATETIME workbook contains no worksheets."), logger)

    logger.info("Loaded %d temporal metric worksheets.", len(excel_data))

    try:
        analysis_text_context = build_workbook_context(excel_data)
    except Exception as exc:
        logger.error("Failed to build temporal workbook context: %s", exc)
        raise CustomException(exc, logger)

    if not analysis_text_context.strip():
        logger.error("NUM-DATETIME workbook produced an empty analysis context.")
        raise CustomException(ValueError("No usable temporal metrics found in workbook."), logger)

    user_prompt = get_numeric_datetime_prompt(analysis_text_context)

    logger.info("Sending temporal workbook data to resilient LLM pipeline...")
    try:
        summary = generate_response(
            system_prompt=SYSTEM_INSTRUCTION_NUM_DATETIME,
            user_prompt=user_prompt,
        )
    except Exception as exc:
        logger.error("NUM-DATETIME LLM summarization failed: %s", exc)
        raise CustomException(exc, logger)

    if not summary or not summary.strip():
        logger.error("LLM returned an empty NUM-DATETIME summary.")
        raise CustomException(ValueError("LLM returned an empty NUM-DATETIME summary."), logger)

    try:
        OUTPUT_TEXT_PATH.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Saving NUM-DATETIME summary to %s", OUTPUT_TEXT_PATH)
        OUTPUT_TEXT_PATH.write_text(summary, encoding="utf-8")
    except Exception as exc:
        logger.error("Failed to save NUM-DATETIME summary: %s", exc)
        raise CustomException(exc, logger)

    logger.info("NUM-DATETIME EDA summary generated successfully.")
    return summary


@track_performance
def execute_bivariate_num_datetime_eda() -> None:
    """Execute NUM-DATETIME EDA workbook summarization."""
    try:
        summary_result = summarize_num_datetime_workbook(WORKBOOK_PATH)
        if summary_result:
            print("\n=== LLM NUMERIC-DATETIME EDA SUMMARY ===\n")
            print(summary_result)
    except Exception as exc:
        logger.exception("NUM-DATETIME EDA summarization execution failed: %s", exc)


if __name__ == "__main__":
    execute_bivariate_num_datetime_eda()