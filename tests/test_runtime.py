import json
from types import SimpleNamespace

import pytest

from multi_agents.runtime import AnalysisRuntime
from multi_agents.workspace import RunWorkspace, WorkspaceLimitError, WorkspaceLimits


class FakeRunner:
    def __init__(self, stop_error=None):
        self.stop_error = stop_error
        self.stopped = 0

    def stop(self):
        self.stopped += 1
        if self.stop_error:
            raise self.stop_error


class FakeExecutor(FakeRunner):
    pass


class FakeGroupChat:
    def __init__(self):
        self.calls = []

    def run(self, dataset_paths, requirement):
        self.calls.append((dataset_paths, requirement))
        return iter(("event",))


def make_workspace(tmp_path, **kwargs):
    return RunWorkspace.create(tmp_path / "temp", tmp_path / "artifacts", **kwargs)


def test_start_cleans_workspace_when_executor_creation_fails(tmp_path, monkeypatch):
    workspace = make_workspace(tmp_path)
    private_root = workspace.code_dir.parent
    monkeypatch.setattr(
        "multi_agents.runtime.create_docker_executor",
        lambda workspace: (_ for _ in ()).throw(RuntimeError("docker unavailable")),
    )

    with pytest.raises(RuntimeError, match="docker unavailable"):
        AnalysisRuntime.start("local-key", workspace)

    assert not workspace.root.exists()
    assert not private_root.exists()


def test_start_stops_executor_when_code_runner_creation_fails(tmp_path, monkeypatch):
    workspace = make_workspace(tmp_path)
    executor = FakeExecutor()
    monkeypatch.setattr("multi_agents.runtime.create_docker_executor", lambda workspace: executor)
    monkeypatch.setattr(
        "multi_agents.runtime.CodeRunner",
        lambda workspace, executor: (_ for _ in ()).throw(RuntimeError("bad runner")),
    )

    with pytest.raises(RuntimeError, match="bad runner"):
        AnalysisRuntime.start("local-key", workspace)

    assert executor.stopped == 1
    assert not workspace.root.exists()


def test_start_cleans_executor_and_workspace_when_group_chat_fails(tmp_path, monkeypatch):
    workspace = make_workspace(tmp_path)
    runner = FakeRunner()
    monkeypatch.setattr("multi_agents.runtime.create_docker_executor", lambda workspace: object())
    monkeypatch.setattr("multi_agents.runtime.CodeRunner", lambda workspace, executor: runner)
    monkeypatch.setattr(
        "multi_agents.runtime.GroupChat",
        lambda api_key, workspace, code_runner: (_ for _ in ()).throw(RuntimeError("bad config")),
    )

    with pytest.raises(RuntimeError, match="bad config"):
        AnalysisRuntime.start("local-key", workspace)

    assert runner.stopped == 1
    assert not workspace.root.exists()


def test_start_preserves_original_error_when_cleanup_fails(tmp_path, monkeypatch):
    workspace = make_workspace(tmp_path)
    runner = FakeRunner(RuntimeError("stop failed"))
    monkeypatch.setattr("multi_agents.runtime.create_docker_executor", lambda workspace: object())
    monkeypatch.setattr("multi_agents.runtime.CodeRunner", lambda workspace, executor: runner)
    monkeypatch.setattr(
        "multi_agents.runtime.GroupChat",
        lambda api_key, workspace, code_runner: (_ for _ in ()).throw(RuntimeError("bad config")),
    )

    with pytest.raises(RuntimeError, match="bad config"):
        AnalysisRuntime.start("local-key", workspace)

    assert runner.stopped == 1
    assert not workspace.root.exists()


def test_run_records_original_prompt_once_before_group_chat(tmp_path):
    workspace = make_workspace(tmp_path)
    group_chat = FakeGroupChat()
    runtime = AnalysisRuntime(workspace, FakeRunner(), group_chat)

    assert list(runtime.run("Find churn drivers.")) == ["event"]

    assert workspace.evidence_dir.joinpath("prompt.txt").read_text() == "Find churn drivers."
    assert group_chat.calls == [((), "Find churn drivers.")]


def test_record_event_retains_trace_and_business_recommendation(tmp_path):
    workspace = make_workspace(tmp_path)
    runtime = AnalysisRuntime(workspace, FakeRunner(), FakeGroupChat())
    event = SimpleNamespace(
        type="text",
        content=SimpleNamespace(
            sender="BusinessTranslator",
            content="Prioritize short-tenure customers.",
        ),
    )

    runtime.record_event(event)

    trace = json.loads(workspace.evidence_dir.joinpath("trace.jsonl").read_text())
    assert trace["type"] == "text"
    assert "Prioritize short-tenure customers." in trace["content"]
    assert workspace.evidence_dir.joinpath("recommendation.txt").read_text() == (
        "Prioritize short-tenure customers."
    )


def test_record_event_does_not_persist_runtime_api_key(tmp_path, monkeypatch):
    workspace = make_workspace(tmp_path)
    runner = FakeRunner()
    group_chat = FakeGroupChat()
    monkeypatch.setattr("multi_agents.runtime.create_docker_executor", lambda workspace: object())
    monkeypatch.setattr("multi_agents.runtime.CodeRunner", lambda workspace, executor: runner)
    monkeypatch.setattr(
        "multi_agents.runtime.GroupChat", lambda api_key, workspace, code_runner: group_chat
    )

    runtime = AnalysisRuntime.start("dummy-local-secret", workspace)
    runtime.record_event(
        SimpleNamespace(type="text", content=SimpleNamespace(sender="Coder", content="done"))
    )

    assert "dummy-local-secret" not in workspace.evidence_dir.joinpath("trace.jsonl").read_text()
    assert not hasattr(runtime, "api_key")


def test_record_event_reports_workspace_limits(tmp_path):
    workspace = make_workspace(tmp_path, limits=WorkspaceLimits(max_record_bytes=8))
    runtime = AnalysisRuntime(workspace, FakeRunner(), FakeGroupChat())

    with pytest.raises(WorkspaceLimitError, match="trace record limit"):
        runtime.record_event(SimpleNamespace(type="text", content="too much trace evidence"))


def test_finish_retains_evidence_when_executor_cleanup_fails(tmp_path):
    workspace = make_workspace(tmp_path)
    runner = FakeRunner(RuntimeError("stop failed"))
    runtime = AnalysisRuntime(workspace, runner, SimpleNamespace())

    runtime.finish("completed")

    retained = workspace.artifacts_root / workspace.run_id
    run = json.loads((retained / "evidence" / "run.json").read_text())
    assert run["failures"] == ["Executor cleanup failed: stop failed"]
    assert runtime.failures == ["Executor cleanup failed: stop failed"]
    assert runner.stopped == 1


def test_finish_is_idempotent(tmp_path):
    workspace = make_workspace(tmp_path)
    runner = FakeRunner()
    runtime = AnalysisRuntime(workspace, runner, SimpleNamespace())

    runtime.finish("completed")
    runtime.finish("failed")

    run = json.loads(
        (workspace.artifacts_root / workspace.run_id / "evidence" / "run.json").read_text()
    )
    assert run["status"] == "completed"
    assert runner.stopped == 1


def test_close_stops_runner_and_removes_temporary_roots_even_when_stop_fails(tmp_path):
    workspace = make_workspace(tmp_path)
    root = workspace.root
    private_root = workspace.code_dir.parent
    runner = FakeRunner(RuntimeError("stop failed"))
    runtime = AnalysisRuntime(workspace, runner, SimpleNamespace())

    runtime.close()
    runtime.close()

    assert runner.stopped == 1
    assert runtime.failures == ["Executor cleanup failed: stop failed"]
    assert not root.exists()
    assert not private_root.exists()
