"""Multivariate EDA workbook summarization using the project's LLM wrapper."""

from pathlib import Path
import pandas as pd
from src.config import BASE_DIR
from src.Utils.exception_handler import CustomException
from src.Utils.llm_variants import generate_response
from src.Utils.logger_setup import get_log, track_performance
from src.EDA_Summary.Prompts.multivariate_prompts import (
    SYSTEM_INSTRUCTION_MULTIVARIATE,
    get_multivariate_prompt,
)


class MultivariateSummarizer:
    """Reads Multivariate EDA Excel metrics and generates an LLM-based executive summary."""

    WORKBOOK_NAME = "multivariate_eda.xlsx"
    SUMMARY_NAME = "multivariate_eda.txt"

    def __init__(self) -> None:
        self.base_dir = BASE_DIR
        self.excel_reports_dir = (
            self.base_dir / "src/EDA_Summary/Multivariate_Analysis/Reports/excel_reports"
        )
        self.text_reports_dir = (
            self.base_dir / "src/EDA_Summary/Multivariate_Analysis/Reports/text_reports"
        )
        self.logs_dir = self.base_dir / "src/EDA_Summary/Multivariate_Analysis/logs"
        self.logger = get_log("MultivariateSummarizer", log_dir=self.logs_dir)

    @track_performance
    def build_workbook_context(self, excel_data: dict[str, pd.DataFrame]) -> str:
        """Convert all workbook worksheets into LLM-readable markdown context."""
        parts = []
        for sheet_name, df in excel_data.items():
            parts.append(
                f"\n\n==============================\n"
                f"METRIC SECTION: {sheet_name.upper()}\n"
                f"==============================\n"
            )
            parts.append(
                "No data available in this metric section.\n"
                if df.empty
                else df.to_markdown(index=False)
            )
        return "".join(parts)

    @track_performance
    def summarize_workbook(self, file_path: Path | str) -> str:
        """Read a Multivariate EDA workbook and generate its summary."""
        file_path = Path(file_path)
        try:
            if not file_path.exists():
                raise FileNotFoundError(f"Multivariate EDA workbook not found: {file_path}")
            if not file_path.is_file():
                raise FileNotFoundError(f"Multivariate EDA path is not a file: {file_path}")

            self.logger.info("Reading Multivariate EDA workbook: %s", file_path)
            excel_data = pd.read_excel(file_path, sheet_name=None)
            if not excel_data:
                raise ValueError("Multivariate EDA workbook contains no worksheets.")

            self.logger.info("Loaded %d Multivariate EDA worksheets.", len(excel_data))
            analysis_text_context = self.build_workbook_context(excel_data)
            if not analysis_text_context.strip():
                raise ValueError("No Multivariate EDA data available for summarization.")

            user_prompt = get_multivariate_prompt(analysis_text_context)
            self.logger.info("Sending Multivariate EDA context to LLM for summarization.")
            summary = generate_response(
                system_prompt=SYSTEM_INSTRUCTION_MULTIVARIATE,
                user_prompt=user_prompt,
            )
            if not summary or not summary.strip():
                raise ValueError("LLM returned an empty Multivariate EDA summary.")

            summary = summary.strip()
            self.text_reports_dir.mkdir(parents=True, exist_ok=True)
            output_path = self.text_reports_dir / self.SUMMARY_NAME
            output_path.write_text(summary, encoding="utf-8")
            self.logger.info("Multivariate EDA summary saved to: %s", output_path)
            return summary

        except Exception as exc:
            self.logger.error("Failed to summarize Multivariate EDA workbook: %s", exc)
            raise CustomException(exc) from exc

    @track_performance
    def execute(self) -> str:
        """Execute summarization using the standardized project workbook location."""
        workbook_path = self.excel_reports_dir / self.WORKBOOK_NAME
        self.logger.info("Executing Multivariate EDA summarization: %s", workbook_path)
        return self.summarize_workbook(workbook_path)

@track_performance
def summarize_multivariate_workbook(file_path: Path | str) -> str:
    """Convenience wrapper for Multivariate EDA summarization."""
    return MultivariateSummarizer().summarize_workbook(file_path)

@track_performance
def execute_multivariate_eda() -> str:
    """Execute summarization using the project's standard workbook path."""
    return MultivariateSummarizer().execute()


if __name__ == "__main__":
    try:
        summary_result = execute_multivariate_eda()
        print("\n=== MULTIVARIATE EDA SUMMARY ===\n")
        print(summary_result)
    except Exception as exc:
        print(f"Multivariate EDA summarization failed: {exc}")