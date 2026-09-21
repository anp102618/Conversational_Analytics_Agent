import json
import time
import pandas as pd
from pathlib import Path
from groq import Groq
from openai import OpenAI
import src
from src.config import config, BASE_DIR

# Import utilities from your utils module
from src.Utils.logger_setup import get_log, track_performance
from src.Utils.exception_handler import CustomException

logger = get_log("EDAClassifier")


class GroqStrategy:
    def __init__(self, api_key: str, model: str):
        self.client = Groq(api_key=api_key)
        self.model = model

    def generate(self, prompt: str, system_instruction: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.1,
        )
        return response.choices[0].message.content


class NvidiaNimFallbackStrategy:
    def __init__(self, api_key: str, model: str, base_url: str = "https://integrate.api.nvidia.com/v1"):
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model

    def generate(self, prompt: str, system_instruction: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.1,
            max_tokens=2048,
        )
        return response.choices[0].message.content


class ResilientGroqContext:
    def __init__(self, primary: GroqStrategy, fallback: NvidiaNimFallbackStrategy, max_retries: int = 3):
        self.primary = primary
        self.fallback = fallback
        self.max_retries = max_retries

    def _execute_with_retry(self, strategy, method_name: str, *args, **kwargs):
        func = getattr(strategy, method_name)
        last_exception = None

        for attempt in range(1, self.max_retries + 1):
            try:
                logger.info(f"Attempt {attempt}/{self.max_retries} using {strategy.__class__.__name__}...")
                return func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                logger.warning(f"Attempt {attempt} failed for {strategy.__class__.__name__}: {e}")
                if attempt < self.max_retries:
                    sleep_time = 2 ** attempt
                    logger.info(f"Retrying in {sleep_time} seconds...")
                    time.sleep(sleep_time)

        raise CustomException(last_exception, logger)

    def generate(self, prompt: str, system_instruction: str) -> str:
        try:
            return self._execute_with_retry(self.primary, "generate", prompt, system_instruction)
        except Exception as primary_err:
            logger.error(f"Primary Groq API exhausted all retries. Failing over to NVIDIA NIM fallback. Details: {primary_err}")
            return self._execute_with_retry(self.fallback, "generate", prompt, system_instruction)


@track_performance
def analyze_eda_columns_with_llm(df: pd.DataFrame) -> dict:
    """Samples a pandas DataFrame and uses Groq (with NVIDIA NIM fallback)

    to classify columns for numeric, categorical, datetime, and EDA types.

    Returns a dictionary of lists.
    """
    try:
        logger.info("Sampling DataFrame for column classification...")
        sample_df = df.sample(n=min(5, len(df)), random_state=42)
        data_sample = sample_df.to_csv(index=False)

        prompt = f"""
        Analyze the following CSV sample of a dataset and its column headers. 
        Classify the columns based on their data types and suitability for Exploratory Data Analysis (EDA).
        
        Data Sample:
        {data_sample}
        
        You must return a valid JSON object ONLY, with no extra text or markdown code formatting blocks outside of the JSON if possible, containing these exact keys as lists of column names:
        - "univariate_candidates": columns best suited for single-variable distribution plots (histograms, boxplots).
        - "bivariate_candidates": pairs or individual columns heavily suited for relationship analysis (scatter plots, correlation).
        - "multivariate_candidates": columns that can be used together in complex multi-variable analysis (e.g., pairplots, heatmaps, cluster analysis).
        - provide the candidates for the above three types of analysis in the form of a dictionary where numeric , categorical and datetime columns are also provided as lists of column names.
        """

        system_instruction = (
            "You are an expert data scientist. Return strictly valid JSON containing a dictionary of lists."
        )

        primary_strategy = GroqStrategy(api_key=config.GROQ_API_KEY, model=config.GROQ_MODEL)
        fallback_strategy = NvidiaNimFallbackStrategy(api_key=config.NVIDIA_API_KEY, model=config.NVIDIA_MODEL)

        resilient_llm = ResilientGroqContext(
            primary=primary_strategy,
            fallback=fallback_strategy,
            max_retries=3
        )

        logger.info("Triggering LLM completion request with fallback support...")
        result_content = resilient_llm.generate(prompt=prompt, system_instruction=system_instruction)
        
        logger.info("Successfully received and parsed classification response.")
        return json.loads(result_content)

    except Exception as e:
        logger.error(f"Failed to analyze EDA columns: {e}")
        raise CustomException(e, logger)


@track_performance
def classify_columns(file_path: Path = BASE_DIR / "src" / "Data" / "cleaned_ecommerce_dataset.csv") -> dict:
    try:
        output_json_path = BASE_DIR / "src" / "Data" / "classified_columns.json"
        
        logger.info(f"Reading dataset from: {file_path}")
        df = pd.read_csv(file_path)
        
        eda_dict = analyze_eda_columns_with_llm(df)
        
        logger.info(f"Saving classified columns to: {output_json_path}")
        output_json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_json_path, "w") as outfile:
            json.dump(eda_dict, outfile, indent=4)
            
        logger.info("Column classification and caching completed successfully.")
        return eda_dict

    except Exception as e:
        logger.error(f"Error occurred during column classification pipeline: {e}")
        raise CustomException(e, logger)


def main() -> None:
    try:
        logger.info("Starting column classification pipeline...")
        result = classify_columns()
        logger.info(f"Pipeline output summary keys: {list(result.keys())}")
    except Exception as e:
        logger.exception(f"Pipeline execution failed: {e}")


if __name__ == "__main__":
    main()