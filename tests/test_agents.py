import inspect
from types import SimpleNamespace

import pytest

import multi_agents.business_analyst as business_analyst_module
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


def test_dataset_registry_rejects_absolute_allowlist_entry(tmp_path):
    workspace_root = tmp_path / "workspace"
    dataset = workspace_root / "datasets" / "sample.csv"
    dataset.parent.mkdir(parents=True)
    dataset.write_bytes(CSV)

    with pytest.raises(ValueError, match="safe relative path within the workspace"):
        DatasetRegistry(workspace_root, (str(dataset),))


def test_dataset_registry_rejects_escaping_allowlist_entry(tmp_path):
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    (tmp_path / "outside.csv").write_bytes(CSV)

    with pytest.raises(ValueError, match="safe relative path within the workspace"):
        DatasetRegistry(workspace_root, ("../outside.csv",))


def test_dataset_registry_rejects_absolute_request_path(tmp_path):
    workspace_root = tmp_path / "workspace"
    dataset = workspace_root / "datasets" / "sample.csv"
    dataset.parent.mkdir(parents=True)
    dataset.write_bytes(CSV)
    registry = DatasetRegistry(workspace_root, ("datasets/sample.csv",))

    with pytest.raises(ValueError, match="not an uploaded dataset"):
        registry.resolve(str(dataset))


def test_dataset_registry_rejects_escaping_request_path(tmp_path):
    workspace_root = tmp_path / "workspace"
    dataset = workspace_root / "datasets" / "sample.csv"
    dataset.parent.mkdir(parents=True)
    dataset.write_bytes(CSV)
    registry = DatasetRegistry(workspace_root, ("datasets/sample.csv",))

    with pytest.raises(ValueError, match="not an uploaded dataset"):
        registry.resolve("../outside.csv")


def test_dataset_registry_rejects_allowlisted_symlink_outside_workspace(tmp_path):
    workspace_root = tmp_path / "workspace"
    dataset_dir = workspace_root / "datasets"
    dataset_dir.mkdir(parents=True)
    outside = tmp_path / "outside.csv"
    outside.write_bytes(CSV)
    (dataset_dir / "linked.csv").symlink_to(outside)

    with pytest.raises(ValueError, match="safe relative path within the workspace"):
        DatasetRegistry(workspace_root, ("datasets/linked.csv",))


def test_dataset_info_bounds_wide_columns_large_cells_and_csv_reads(monkeypatch, tmp_path):
    workspace_root = tmp_path / "workspace"
    dataset = workspace_root / "datasets" / "wide.csv"
    dataset.parent.mkdir(parents=True)
    long_column_name = "column_" + "n" * 500
    large_cell = "x" * 20_000
    column_names = [long_column_name, *(f"numeric_{index}" for index in range(24))]
    first_row = [large_cell, *(str(index) for index in range(24))]
    second_row = ["category", *(str(index + 1) for index in range(24))]
    dataset.write_text(
        ",".join(column_names) + "\n" + ",".join(first_row) + "\n" + ",".join(second_row) + "\n"
    )
    registry = DatasetRegistry(workspace_root, ("datasets/wide.csv",))
    read_calls = []
    real_read_csv = business_analyst_module.pd.read_csv

    def tracked_read_csv(*args, **kwargs):
        read_calls.append(kwargs)
        return real_read_csv(*args, **kwargs)

    monkeypatch.setattr(business_analyst_module.pd, "read_csv", tracked_read_csv)

    message = registry.get_data_info("datasets/wide.csv").message

    assert len(message) <= 8_000
    assert large_cell[:81] not in message
    assert long_column_name[:81] not in message
    assert "15 additional columns omitted" in message
    assert "sample-inferred" in message
    assert "Shape: 2 rows × 25 columns" in message
    assert all("nrows" in call or "chunksize" in call for call in read_calls)


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


def test_group_chat_run_passes_exact_message_round_limit_and_returns_iterator(monkeypatch):
    pattern = object()
    events = [SimpleNamespace(type="first"), SimpleNamespace(type="second")]
    calls = []

    def fake_run_group_chat(*, pattern, messages, max_rounds):
        calls.append(
            {
                "pattern": pattern,
                "messages": messages,
                "max_rounds": max_rounds,
            }
        )
        return SimpleNamespace(events=events)

    monkeypatch.setattr(group_chat_module, "run_group_chat", fake_run_group_chat)
    group = GroupChat.__new__(GroupChat)
    group.pattern = pattern

    result = group.run(
        ("datasets/first.csv", "datasets/second.csv"),
        "Compare outcomes.",
    )

    assert calls == [
        {
            "pattern": pattern,
            "messages": (
                "Data paths: ['datasets/first.csv', 'datasets/second.csv']\n"
                "Requirements: Compare outcomes."
            ),
            "max_rounds": 60,
        }
    ]
    assert iter(result) is result
    assert list(result) == events


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
