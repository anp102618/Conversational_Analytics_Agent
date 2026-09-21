# src/eda/summarize_categorical.py

from pathlib import Path
import pandas as pd

from src.config import BASE_DIR, config
from src.Utils.logger_setup import get_log, track_performance
from src.Utils.exception_handler import CustomException

# Import your helper functions and categorical prompt templates
from src.Utils.llm_variants import generate_response
from src.EDA_Summary.Prompts.univariate_prompts import get_univariate_cat_prompt, SYSTEM_INSTRUCTION_CAT

logger = get_log("UnivariateCATSummarizer", log_dir =BASE_DIR / "src" / "EDA_Summary" / "Univariate_Analysis" / "logs")


@track_performance
def summarize_univariate_cat_workbook(file_path: Path | str) -> str:
    """Reads worksheets containing Categorical Univariate EDA metrics from an Excel workbook, summarizes them using the resilient LLM strategy, and saves the text report."""
    
    file_path = Path(file_path)
    if not file_path.exists():
        logger.error(f"File not found: {file_path}")
        raise CustomException(FileNotFoundError(f"File not found: {file_path}"), logger)

    try:
        logger.info(f"Reading Excel workbook from {file_path}...")
        excel_data = pd.read_excel(file_path, sheet_name=None)
    except Exception as e:
        logger.error(f"Error reading Excel file: {e}")
        raise CustomException(e, logger)

    analysis_text_context = ""
    for sheet_name, df in excel_data.items():
        analysis_text_context += "\n\n==============================\n"
        analysis_text_context += f"METRIC SECTION: {sheet_name.upper()}\n"
        analysis_text_context += "==============================\n"
        analysis_text_context += df.to_markdown(index=False)

    # Generate prompt using the isolated categorical template
    user_prompt = get_univariate_cat_prompt(analysis_text_context)

    logger.info("Sending categorical univariate workbook data to resilient LLM pipeline for summarization...")
    try:
        # Utilize the resilient generate_response wrapper (Nvidia primary -> Gemini fallback)
        summary = generate_response(system_prompt=SYSTEM_INSTRUCTION_CAT, user_prompt=user_prompt)
        
        # Define output text file path and ensure directory exists
        output_txt_path = BASE_DIR / "src" / "EDA_Summary" / "Univariate_Analysis" / "Reports" / "text_reports" / "categorical_uni_eda.txt"
        output_txt_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save summary to text file
        logger.info(f"Saving summary report to {output_txt_path}...")
        output_txt_path.write_text(summary, encoding="utf-8")
        
        logger.info("Successfully generated and saved categorical univariate summary.")
        return summary
    except Exception as e:
        logger.error(f"LLM summarization or saving failed: {e}")
        raise CustomException(e, logger)

@track_performance
def execute_univariate_cat_eda() -> None:
    output_workbook_path = Path(BASE_DIR / "src" / "EDA_Summary" / "Univariate_Analysis" / "Reports" / "excel_reports" / "categorical_uni_eda.xlsx")
    
    try:
        summary_result = summarize_univariate_cat_workbook(output_workbook_path)
        if summary_result:
            print("\n=== LLM CATEGORICAL UNIVARIATE EDA SUMMARY ===\n")
            print(summary_result)
    except Exception as e:
        logger.exception(f"Execution failed: {e}")


if __name__ == "__main__":
    execute_univariate_cat_eda()