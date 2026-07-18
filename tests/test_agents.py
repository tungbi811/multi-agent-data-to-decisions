import inspect
from types import SimpleNamespace

import pytest

import multi_agents.group_chat as group_chat_module
from multi_agents.business_analyst import BusinessAnalyst, DatasetRegistry
from multi_agents.config import DEFAULT_MODEL, get_model_name
from multi_agents.group_chat import MAX_ROUNDS, GroupChat
from multi_agents.workspace import RunWorkspace


CSV = b"feature,target\n1,0\n2,1\n"


def test_model_name_has_one_default_and_environment_override(monkeypatch):
    monkeypatch.delenv("AUTO_DS_MODEL", raising=False)
    assert DEFAULT_MODEL == "gpt-4.1-mini"
    assert get_model_name() == "gpt-4.1-mini"
    monkeypatch.setenv("AUTO_DS_MODEL", "gpt-4.1-nano")
    assert get_model_name() == "gpt-4.1-nano"


def test_dataset_registry_allows_only_uploaded_relative_paths(tmp_path):
    workspace = RunWorkspace.create(tmp_path / "temp", tmp_path / "artifacts")
    allowed = workspace.save_upload("sample.csv", CSV)
    registry = DatasetRegistry(workspace.root, workspace.dataset_relative_paths)
    assert registry.resolve("datasets/sample.csv") == allowed.resolve()
    with pytest.raises(ValueError, match="not an uploaded dataset"):
        registry.resolve("../README.md")


def test_business_analyst_prompt_names_registered_tool_and_real_target():
    source = inspect.getsource(BusinessAnalyst)
    assert "complete_business_analyst" in source
    assert "complete_business_analysis_task" not in source
    assert "DataExplorer" not in source
    assert "BusinessTranslator" in source


def test_group_chat_uses_bounded_rounds_and_injected_runtime():
    assert MAX_ROUNDS == 60
    parameters = inspect.signature(GroupChat).parameters
    assert tuple(parameters) == ("api_key", "workspace", "code_runner")


def test_group_chat_constructs_ordered_agents_with_supplied_code_runner(monkeypatch, tmp_path):
    created = {}
    config_calls = []
    after_work_target = object()

    class DummyHandoffs:
        def set_after_work(self, target):
            created["after_work_target"] = target

    class DummyAgent:
        def __init__(self, name, llm_config, **dependencies):
            self.name = name
            self.llm_config = llm_config
            self.handoffs = DummyHandoffs()
            for key, value in dependencies.items():
                setattr(self, key, value)
            created[name] = self

    def build_config(api_key, temperature):
        config_calls.append((api_key, temperature))
        return SimpleNamespace(temperature=temperature)

    monkeypatch.setattr(group_chat_module, "build_llm_config", build_config)
    monkeypatch.setattr(
        group_chat_module,
        "BusinessAnalyst",
        lambda registry, llm_config: DummyAgent("BusinessAnalyst", llm_config, registry=registry),
    )
    monkeypatch.setattr(
        group_chat_module,
        "BusinessTranslator",
        lambda llm_config: DummyAgent("BusinessTranslator", llm_config),
    )
    monkeypatch.setattr(
        group_chat_module,
        "DataScientist",
        lambda llm_config: DummyAgent("DataScientist", llm_config),
    )
    monkeypatch.setattr(
        group_chat_module,
        "Coder",
        lambda code_runner, llm_config: DummyAgent("Coder", llm_config, code_runner=code_runner),
    )
    monkeypatch.setattr(
        group_chat_module,
        "UserProxyAgent",
        lambda **kwargs: SimpleNamespace(**kwargs),
    )
    monkeypatch.setattr(
        group_chat_module,
        "DefaultPattern",
        lambda **kwargs: SimpleNamespace(**kwargs),
    )
    monkeypatch.setattr(group_chat_module, "RevertToUserTarget", lambda: after_work_target)
    monkeypatch.setattr(
        group_chat_module,
        "AgentTarget",
        lambda agent: SimpleNamespace(agent=agent),
    )

    workspace = RunWorkspace.create(tmp_path / "temp", tmp_path / "artifacts")
    code_runner = object()
    group = GroupChat("dummy-api-key", workspace, code_runner)

    assert created["Coder"].code_runner is code_runner
    assert [agent.name for agent in group.pattern.agents] == [
        "BusinessAnalyst",
        "BusinessTranslator",
        "DataScientist",
        "Coder",
    ]
    assert group.pattern.initial_agent is created["BusinessAnalyst"]
    assert created["after_work_target"] is after_work_target
    assert config_calls == [
        ("dummy-api-key", 0.5),
        ("dummy-api-key", 0.3),
        ("dummy-api-key", 0.3),
        ("dummy-api-key", 0.0),
    ]
