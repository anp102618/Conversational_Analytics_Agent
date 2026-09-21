import time
from abc import ABC, abstractmethod
from typing import Generator, Any
from openai import OpenAI
from google import genai
from google.genai import types

# Assuming your previous utilities are imported from your utils module
from src.Utils.logger_setup import get_log, track_performance
from src.Utils.exception_handler import CustomException

logger = get_log()


# --- 1. Abstract Strategy Interface ---
class LLMStrategy(ABC):
    @abstractmethod
    def generate(self, prompt: str, system_instruction: str = None) -> str:
        """Generates a complete response (Unary)."""
        pass

    @abstractmethod
    def generate_stream(self, prompt: str, system_instruction: str = None) -> Generator[str, None, None]:
        """Generates a response stream chunk by chunk."""
        pass


# --- 2. Concrete Strategy: NVIDIA NIM ---
class NvidiaNimStrategy(LLMStrategy):
    def __init__(self, api_key: str, model: str, base_url: str = "https://integrate.api.nvidia.com/v1"):
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model

    @track_performance
    def generate(self, prompt: str, system_instruction: str = None) -> str:
        try:
            messages = []
            if system_instruction:
                messages.append({"role": "system", "content": system_instruction})
            messages.append({"role": "user", "content": prompt})

            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.2,
                max_tokens=2048,
            )
            return response.choices[0].message.content
        except Exception as e:
            raise CustomException(e, logger)

    @track_performance
    def generate_stream(self, prompt: str, system_instruction: str = None) -> Generator[str, None, None]:
        try:
            messages = []
            if system_instruction:
                messages.append({"role": "system", "content": system_instruction})
            messages.append({"role": "user", "content": prompt})

            stream = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.2,
                max_tokens=2048,
                stream=True,
            )
            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            raise CustomException(e, logger)


# --- 3. Concrete Strategy: Gemini Flash-Lite ---
class GeminiFlashLiteStrategy(LLMStrategy):
    def __init__(self, api_key: str, model: str = "gemini-3.1-flash-lite"):
        self.client = genai.Client(api_key=api_key)
        self.model = model

    @track_performance
    def generate(self, prompt: str, system_instruction: str = None) -> str:
        try:
            config_kwargs = {"temperature": 0.2, "max_output_tokens": 2048}
            if system_instruction:
                config_kwargs["system_instruction"] = system_instruction

            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=types.GenerateContentConfig(**config_kwargs)
            )
            return response.text
        except Exception as e:
            raise CustomException(e, logger)

    @track_performance
    def generate_stream(self, prompt: str, system_instruction: str = None) -> Generator[str, None, None]:
        try:
            config_kwargs = {"temperature": 0.2, "max_output_tokens": 2048}
            if system_instruction:
                config_kwargs["system_instruction"] = system_instruction

            response = self.client.models.generate_content_stream(
                model=self.model,
                contents=prompt,
                config=types.GenerateContentConfig(**config_kwargs)
            )
            for chunk in response:
                if chunk.text:
                    yield chunk.text
        except Exception as e:
            raise CustomException(e, logger)


# --- 4. Context / Failover Wrapper with Retries ---
class ResilientLLMContext:
    def __init__(self, primary_llm: LLMStrategy, fallback_llm: LLMStrategy, max_retries: int = 3):
        self.primary_llm = primary_llm
        self.fallback_llm = fallback_llm
        self.max_retries = max_retries

    def _execute_with_retry(self, strategy: LLMStrategy, method_name: str, *args, **kwargs) -> Any:
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

    def generate(self, prompt: str, system_instruction: str = None) -> str:
        try:
            return self._execute_with_retry(self.primary_llm, "generate", prompt, system_instruction)
        except Exception as primary_err:
            logger.error(f"Primary LLM exhausted all retries. Failing over to fallback. Details: {primary_err}")
            return self._execute_with_retry(self.fallback_llm, "generate", prompt, system_instruction)

    def generate_stream(self, prompt: str, system_instruction: str = None) -> Generator[str, None, None]:
        try:
            yield from self._execute_with_retry(self.primary_llm, "generate_stream", prompt, system_instruction)
        except Exception as primary_err:
            logger.error(f"Primary streaming failed completely. Failing over to fallback stream. Details: {primary_err}")
            yield from self._execute_with_retry(self.fallback_llm, "generate_stream", prompt, system_instruction)


# --- Execution Entrypoint ---
def generate_response(system_prompt: str, user_prompt: str) -> str:
    from src.config import config

    nvidia_strategy = NvidiaNimStrategy(
        api_key=config.NVIDIA_API_KEY,
        model=config.NVIDIA_MODEL
    )
    
    gemini_strategy = GeminiFlashLiteStrategy(
        api_key=config.GEMINI_API_KEY,
        model="gemini-3.1-flash-lite"
    )

    llm = ResilientLLMContext(
        primary_llm=nvidia_strategy,
        fallback_llm=gemini_strategy,
        max_retries=3
    )

    print("\n--- Testing Unary Response ---")
    final_response = llm.generate(prompt=user_prompt, system_instruction=system_prompt)
    print(final_response)

    return final_response


def generate_stream_response(system_prompt: str, user_prompt: str) -> Generator[str, None, None]:
    from src.config import config

    nvidia_strategy = NvidiaNimStrategy(
        api_key=config.NVIDIA_API_KEY,
        model=config.NVIDIA_MODEL
    )
    
    gemini_strategy = GeminiFlashLiteStrategy(
        api_key=config.GEMINI_API_KEY,
        model="gemini-3.1-flash-lite"
    )

    llm = ResilientLLMContext(
        primary_llm=nvidia_strategy,
        fallback_llm=gemini_strategy,
        max_retries=3
    )

    print("\n--- Testing Streaming Response ---")
    for chunk in llm.generate_stream(prompt=user_prompt, system_instruction=system_prompt):
        print(chunk, end="", flush=True)
        yield chunk
    print()