from pathlib import Path
from src.config import BASE_DIR


REPORT_DIRS = {
    "Univariate EDA": BASE_DIR / "src/EDA_Summary/Univariate_Analysis/Reports/text_reports",
    "Bivariate EDA": BASE_DIR / "src/EDA_Summary/Bivariate_Analysis/Reports/text_reports",
    "Multivariate EDA": BASE_DIR / "src/EDA_Summary/Multivariate_Analysis/Reports/text_reports",
}

OUTPUT_DIR = BASE_DIR / "src/Data/EDA_Reports"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def merge_text_reports(input_dir: Path, output_path: Path, report_name: str):
    files = sorted(input_dir.glob("*.txt"))

    if not files:
        print(f"[WARNING] No text files found: {input_dir}")
        return

    with output_path.open("w", encoding="utf-8") as out:
        out.write(f"# {report_name}\n\n")

        for file in files:
            content = file.read_text(encoding="utf-8").strip()
            if not content:
                continue

            out.write(f"\n{'=' * 80}\n")
            out.write(f"## {file.stem}\n")
            out.write(f"{'=' * 80}\n\n")
            out.write(content + "\n\n")

    print(f"[CREATED] {output_path}")

def eda_text_report_generation():
    for name, input_dir in REPORT_DIRS.items():
        filename = name.split()[0].lower() + ".txt"
        merge_text_reports(input_dir, OUTPUT_DIR / filename, name)

    print("\nEDA report generation completed.")
