import os

from autogen import LLMConfig


DEFAULT_MODEL = "gpt-4.1-mini"


def get_model_name() -> str:
    return os.environ.get("AUTO_DS_MODEL", DEFAULT_MODEL)


def build_llm_config(api_key: str, temperature: float) -> LLMConfig:
    return LLMConfig(
        api_type="openai",
        api_key=api_key,
        model=get_model_name(),
        temperature=temperature,
        stream=False,
        parallel_tool_calls=False,
    )
