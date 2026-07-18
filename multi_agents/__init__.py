from importlib import import_module


_EXPORTS = {
    "BusinessAnalyst": (".business_analyst", "BusinessAnalyst"),
    "BusinessTranslator": (".business_translator", "BusinessTranslator"),
    "Coder": (".coder", "Coder"),
    "DataScientist": (".data_scientist", "DataScientist"),
    "DatasetRegistry": (".business_analyst", "DatasetRegistry"),
    "GroupChat": (".group_chat", "GroupChat"),
    "build_llm_config": (".config", "build_llm_config"),
    "get_model_name": (".config", "get_model_name"),
}

__all__ = [
    "BusinessAnalyst",
    "BusinessTranslator",
    "Coder",
    "DataScientist",
    "DatasetRegistry",
    "GroupChat",
    "build_llm_config",
    "get_model_name",
]


def __getattr__(name: str):
    try:
        module_name, attribute_name = _EXPORTS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None

    value = getattr(import_module(module_name, __name__), attribute_name)
    globals()[name] = value
    return value
