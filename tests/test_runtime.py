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


def test_start_preserves_original_error_when_workspace_cleanup_fails(tmp_path, monkeypatch):
    workspace = make_workspace(tmp_path)
    runner = FakeRunner()
    monkeypatch.setattr("multi_agents.runtime.create_docker_executor", lambda workspace: object())
    monkeypatch.setattr("multi_agents.runtime.CodeRunner", lambda workspace, executor: runner)
    monkeypatch.setattr(
        "multi_agents.runtime.GroupChat",
        lambda api_key, workspace, code_runner: (_ for _ in ()).throw(RuntimeError("bad config")),
    )
    monkeypatch.setattr(
        workspace,
        "close",
        lambda: (_ for _ in ()).throw(OSError("cleanup secret must-not-mask")),
    )

    with pytest.raises(RuntimeError, match="bad config"):
        AnalysisRuntime.start("local-key", workspace)

    assert runner.stopped == 1


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


def test_record_event_normalizes_malformed_trace_fields(tmp_path):
    workspace = make_workspace(tmp_path)
    runtime = AnalysisRuntime(workspace, FakeRunner(), FakeGroupChat())

    runtime.record_event(
        SimpleNamespace(
            type="text",
            content=SimpleNamespace(
                sender=["Coder"],
                content={"api_key": "must-not-leak"},
            ),
        )
    )

    trace = json.loads(workspace.evidence_dir.joinpath("trace.jsonl").read_text())
    assert trace == {
        "type": "text",
        "sender": "System",
        "content": "Malformed event content.",
    }
    assert "must-not-leak" not in workspace.evidence_dir.joinpath("trace.jsonl").read_text()


def test_record_failure_is_bounded_deduplicated_and_redacts_secret_like_values(tmp_path):
    workspace = make_workspace(tmp_path)
    runtime = AnalysisRuntime(workspace, FakeRunner(), FakeGroupChat())
    reason = "Iterator failed; api_key=must-not-leak; " + "x" * 2_000

    runtime.record_failure(reason)
    runtime.record_failure(reason)

    assert len(runtime.failures) == 1
    assert len(runtime.failures[0]) <= 1_000
    assert "must-not-leak" not in runtime.failures[0]
    assert "[REDACTED]" in runtime.failures[0]


def test_record_failure_is_retained_in_final_run_evidence(tmp_path):
    workspace = make_workspace(tmp_path)
    runtime = AnalysisRuntime(workspace, FakeRunner(), FakeGroupChat())

    runtime.record_failure("Event iterator failed.")
    runtime.finish("failed")

    retained = workspace.artifacts_root / workspace.run_id
    run = json.loads(retained.joinpath("evidence", "run.json").read_text())
    assert run["failures"] == ["Event iterator failed."]


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


def test_close_keeps_runtime_retryable_when_workspace_cleanup_fails(tmp_path, monkeypatch):
    workspace = make_workspace(tmp_path)
    runner = FakeRunner()
    runtime = AnalysisRuntime(workspace, runner, SimpleNamespace())
    original_close = workspace.close
    attempts = 0

    def fail_once():
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise OSError("temporary cleanup failure")
        original_close()

    monkeypatch.setattr(workspace, "close", fail_once)

    with pytest.raises(OSError, match="temporary cleanup failure"):
        runtime.close()

    assert runtime.finished is False
    assert runner.stopped == 1
    assert workspace.root.exists()

    runtime.close()

    assert runtime.finished is True
    assert runner.stopped == 1
    assert not workspace.root.exists()


def test_finish_keeps_runtime_retryable_when_workspace_finalization_fails(tmp_path, monkeypatch):
    workspace = make_workspace(tmp_path)
    runner = FakeRunner()
    runtime = AnalysisRuntime(workspace, runner, SimpleNamespace())
    original_finalize = workspace.finalize
    attempts = 0

    def fail_once(status, elapsed, failures):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise OSError("temporary finalize failure")
        return original_finalize(status, elapsed, failures)

    monkeypatch.setattr(workspace, "finalize", fail_once)

    with pytest.raises(OSError, match="temporary finalize failure"):
        runtime.finish("failed")

    assert runtime.finished is False
    assert runner.stopped == 1
    assert workspace.root.exists()

    runtime.finish("failed")

    retained = workspace.artifacts_root / workspace.run_id
    assert runtime.finished is True
    assert runner.stopped == 1
    assert retained.joinpath("evidence", "run.json").is_file()


def test_close_retries_pending_finalization_instead_of_discarding_evidence(tmp_path, monkeypatch):
    workspace = make_workspace(tmp_path)
    runner = FakeRunner()
    runtime = AnalysisRuntime(workspace, runner, SimpleNamespace())
    original_finalize = workspace.finalize
    attempts = 0

    def fail_once(status, elapsed, failures):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise OSError("temporary finalize failure")
        return original_finalize(status, elapsed, failures)

    monkeypatch.setattr(workspace, "finalize", fail_once)

    with pytest.raises(OSError, match="temporary finalize failure"):
        runtime.finish("completed")
    runtime.close()

    retained = workspace.artifacts_root / workspace.run_id
    run = json.loads(retained.joinpath("evidence", "run.json").read_text())
    assert run["status"] == "completed"
    assert runner.stopped == 1


def test_finalization_retry_reuses_first_terminal_elapsed_time(tmp_path, monkeypatch):
    workspace = make_workspace(tmp_path)
    runner = FakeRunner()
    monotonic_values = iter((10.0, 15.0, 100.0))
    monkeypatch.setattr("multi_agents.runtime.time.monotonic", lambda: next(monotonic_values))
    runtime = AnalysisRuntime(workspace, runner, SimpleNamespace())
    original_finalize = workspace.finalize
    finalize_calls = []

    def fail_once(status, elapsed, failures):
        finalize_calls.append((status, elapsed))
        if len(finalize_calls) == 1:
            raise OSError("temporary finalize failure")
        return original_finalize(status, elapsed, failures)

    monkeypatch.setattr(workspace, "finalize", fail_once)

    with pytest.raises(OSError, match="temporary finalize failure"):
        runtime.finish("completed")
    runtime.finish("failed")

    assert finalize_calls == [("completed", 5.0), ("completed", 5.0)]
    assert runner.stopped == 1
    retained = workspace.artifacts_root / workspace.run_id
    run = json.loads(retained.joinpath("evidence", "run.json").read_text())
    assert run["status"] == "completed"
    assert run["runtime_seconds"] == 5.0
