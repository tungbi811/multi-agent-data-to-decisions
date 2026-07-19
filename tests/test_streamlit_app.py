import ast
from pathlib import Path
from types import SimpleNamespace

from streamlit.testing.v1 import AppTest


ROOT = Path(__file__).parents[1]


class FakeRuntime:
    def __init__(self, run_events=()):
        self.events = []
        self.finished = []
        self.closed = 0
        self.failures = []
        self.actions = []
        self.run_events = run_events

    def run(self, requirement):
        return iter(self.run_events)

    def record_event(self, event):
        self.events.append(event)

    def record_failure(self, reason):
        self.failures.append(reason)
        self.actions.append(("failure", reason))

    def finish(self, status):
        self.finished.append(status)
        self.actions.append(("finish", status))

    def close(self):
        self.closed += 1


class CleanupFailRuntime(FakeRuntime):
    def close(self):
        self.closed += 1
        raise OSError("cleanup failed with api_key=must-not-leak")


class FinishFailRuntime(FakeRuntime):
    def finish(self, status):
        self.finished.append(status)
        raise OSError("finalize failed with api_key=must-not-leak")


class RetryCleanupRuntime(FakeRuntime):
    def close(self):
        self.closed += 1
        if self.closed == 1:
            raise OSError("temporary cleanup failure")


class FailingRunRuntime(FakeRuntime):
    def run(self, requirement):
        raise RuntimeError("event iterator unavailable")


class EvidenceLimitRuntime(FakeRuntime):
    def record_event(self, event):
        raise RuntimeError("Workspace trace count limit exceeded.")


class FailingIterator:
    def __iter__(self):
        return self

    def __next__(self):
        raise RuntimeError("event source failed")


def configure_analysis(app, tmp_path):
    upload = tmp_path / "task-five-review.csv"
    upload.write_text("customer_id,churn\n1,0\n")
    app.text_input[0].input("dummy-local-key")
    app.file_uploader[0].upload(upload.name, upload.read_bytes(), "text/csv")
    app.text_area[0].input("Analyze churn.")
    return upload


def test_page_config_is_the_first_streamlit_call():
    tree = ast.parse((ROOT / "main.py").read_text())
    streamlit_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "st"
    ]

    assert min(streamlit_calls, key=lambda node: node.lineno).func.attr == "set_page_config"


def test_app_starts_without_api_key_or_docker(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    app = AppTest.from_file(str(ROOT / "main.py")).run(timeout=10)

    assert not app.exception
    assert app.title[0].value == "🤖 From Data to Decisions: A Multi-Agent System"


def test_sidebar_never_prefills_key_from_process_environment(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-reach-widget")

    app = AppTest.from_file(str(ROOT / "main.py")).run(timeout=10)

    assert not app.exception
    assert app.text_input[0].value == ""
    assert "os.environ" not in (ROOT / "utils" / "sidebar.py").read_text()


def test_sidebar_keeps_uploads_in_memory_only():
    tree = ast.parse((ROOT / "utils" / "sidebar.py").read_text())
    called_names = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assigned_attributes = {
        target.attr
        for node in ast.walk(tree)
        if isinstance(node, (ast.Assign, ast.AnnAssign))
        for target in (node.targets if isinstance(node, ast.Assign) else (node.target,))
        if isinstance(target, ast.Attribute)
    }

    assert "open" not in called_names
    assert "dataset_paths" not in assigned_attributes
    assert "uploaded_files" in assigned_attributes


def test_ui_key_is_never_assigned_to_process_environment():
    tree = ast.parse((ROOT / "main.py").read_text())
    environment_writes = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign))
        for target in (node.targets if isinstance(node, ast.Assign) else (node.target,))
        if isinstance(target, ast.Subscript)
        and isinstance(target.value, ast.Attribute)
        and isinstance(target.value.value, ast.Name)
        and target.value.value.id == "os"
        and target.value.attr == "environ"
    ]

    assert environment_writes == []


def test_run_failure_closes_started_runtime_and_keeps_upload_out_of_repository(
    tmp_path, monkeypatch
):
    upload = tmp_path / "task-five-unique.csv"
    upload.write_text("customer_id,churn\n1,0\n")
    runtime = FailingRunRuntime()
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(
        "agents.runtime.AnalysisRuntime.start",
        lambda api_key, workspace: runtime,
    )
    app = AppTest.from_file(str(ROOT / "main.py")).run(timeout=10)
    app.text_input[0].input("dummy-local-key")
    app.file_uploader[0].upload(upload.name, upload.read_bytes(), "text/csv")
    app.text_area[0].input("Analyze churn.")

    app.button(key="run_analysis").click().run(timeout=10)

    assert not app.exception
    assert runtime.closed == 1
    assert not (ROOT / "data" / "uploads" / upload.name).exists()


def test_completed_run_can_start_and_finalize_a_second_analysis(tmp_path, monkeypatch):
    completion = SimpleNamespace(type="run_completion", content="Analysis complete.")
    first = FakeRuntime((completion,))
    second = FakeRuntime((completion,))
    runtimes = iter((first, second))
    monkeypatch.setattr(
        "agents.runtime.AnalysisRuntime.start",
        lambda api_key, workspace: next(runtimes),
    )
    app = AppTest.from_file(str(ROOT / "main.py")).run(timeout=10)
    configure_analysis(app, tmp_path)

    app.button(key="run_analysis").click().run(timeout=10)
    app.button(key="run_analysis").click().run(timeout=10)

    assert not app.exception
    assert first.finished == ["completed"]
    assert second.finished == ["completed"]
    assert second.events == [completion]
    assert app.session_state.runtime is None


def test_new_analysis_closes_active_runtime_and_resets_run_state(tmp_path, monkeypatch):
    completion = SimpleNamespace(type="run_completion", content="Second complete.")
    old_runtime = FakeRuntime()
    new_runtime = FakeRuntime((completion,))
    monkeypatch.setattr(
        "agents.runtime.AnalysisRuntime.start",
        lambda api_key, workspace: new_runtime,
    )
    app = AppTest.from_file(str(ROOT / "main.py")).run(timeout=10)
    configure_analysis(app, tmp_path)
    app.session_state.runtime = old_runtime
    app.session_state.events = iter((SimpleNamespace(type="input_request", content=None),))
    app.session_state.event = object()
    app.session_state.terminated = True
    app.session_state.awaiting_response = True
    app.session_state.user_input = "stale"
    app.session_state.last_agent_name = "Coder"

    app.button(key="run_analysis").click().run(timeout=10)

    assert not app.exception
    assert old_runtime.closed == 1
    assert new_runtime.finished == ["completed"]
    assert new_runtime.events == [completion]
    assert app.session_state.awaiting_response is False
    assert app.session_state.last_agent_name is None


def test_active_runtime_cleanup_failure_aborts_new_analysis(tmp_path, monkeypatch):
    old_runtime = CleanupFailRuntime()
    starts = []
    monkeypatch.setattr(
        "agents.runtime.AnalysisRuntime.start",
        lambda api_key, workspace: starts.append(workspace),
    )
    app = AppTest.from_file(str(ROOT / "main.py")).run(timeout=10)
    configure_analysis(app, tmp_path)
    app.session_state.runtime = old_runtime

    app.button(key="run_analysis").click().run(timeout=10)

    assert not app.exception
    assert starts == []
    assert app.session_state.runtime is old_runtime
    assert all("must-not-leak" not in error.value for error in app.error)


def test_run_completion_finishes_runtime():
    app = AppTest.from_file(str(ROOT / "main.py")).run(timeout=10)
    runtime = FakeRuntime()
    app.session_state.runtime = runtime
    app.session_state.events = iter(
        (SimpleNamespace(type="run_completion", content="Analysis complete."),)
    )

    app.run(timeout=10)

    assert not app.exception
    assert runtime.finished == ["completed"]
    assert app.session_state.runtime is None


def test_iterator_exhaustion_finishes_runtime_as_failed():
    app = AppTest.from_file(str(ROOT / "main.py")).run(timeout=10)
    runtime = FakeRuntime()
    app.session_state.runtime = runtime
    app.session_state.events = iter(())

    app.run(timeout=10)

    assert not app.exception
    assert runtime.finished == ["failed"]
    assert app.session_state.terminated is True
    assert runtime.actions[0] == ("failure", "Event stream ended before completion.")
    assert runtime.actions[1] == ("finish", "failed")


def test_iterator_failure_finishes_runtime_without_uncaught_exception():
    app = AppTest.from_file(str(ROOT / "main.py")).run(timeout=10)
    runtime = FakeRuntime()
    app.session_state.runtime = runtime
    app.session_state.events = FailingIterator()

    app.run(timeout=10)

    assert not app.exception
    assert runtime.finished == ["failed"]
    assert app.session_state.terminated is True
    assert runtime.failures == ["Event iterator failed."]


def test_malformed_coder_event_becomes_a_system_message():
    app = AppTest.from_file(str(ROOT / "main.py")).run(timeout=10)
    runtime = FakeRuntime()
    malformed = SimpleNamespace(
        type="tool_call",
        content=SimpleNamespace(
            sender="Coder",
            tool_calls=[SimpleNamespace(function=SimpleNamespace(arguments="not-json"))],
        ),
    )
    app.session_state.runtime = runtime
    app.session_state.events = iter((malformed,))

    app.run(timeout=10)

    assert not app.exception
    assert any(
        message["role"] == "System" and "Malformed Coder tool call" in message["content"]
        for message in app.session_state.messages
    )


def test_malformed_known_text_payload_is_normalized_before_rendering():
    app = AppTest.from_file(str(ROOT / "main.py")).run(timeout=10)
    runtime = FakeRuntime()
    app.session_state.runtime = runtime
    app.session_state.events = iter(
        (
            SimpleNamespace(
                type="text",
                content=SimpleNamespace(
                    sender=["Coder"],
                    content={"api_key": "must-not-leak"},
                ),
            ),
        )
    )

    app.run(timeout=10)

    assert not app.exception
    assert {"role": "System", "content": "Malformed event content."} in (app.session_state.messages)
    assert "must-not-leak" not in str(app.session_state.messages)


def test_malformed_tool_sender_and_response_are_normalized_before_rendering():
    app = AppTest.from_file(str(ROOT / "main.py")).run(timeout=10)
    runtime = FakeRuntime()
    app.session_state.runtime = runtime
    app.session_state.last_agent_name = ["Coder"]
    app.session_state.events = iter(
        (
            SimpleNamespace(
                type="tool_response",
                content=SimpleNamespace(content={"secret": "must-not-leak"}),
            ),
        )
    )

    app.run(timeout=10)

    assert not app.exception
    assert any(
        message["role"] == "System" and message["content"] == "Malformed event content."
        for message in app.session_state.messages
    )
    assert "must-not-leak" not in str(app.session_state.messages)


def test_malformed_tool_call_sender_is_rejected_without_leaking_payload():
    app = AppTest.from_file(str(ROOT / "main.py")).run(timeout=10)
    runtime = FakeRuntime()
    app.session_state.runtime = runtime
    app.session_state.events = iter(
        (
            SimpleNamespace(
                type="tool_call",
                content=SimpleNamespace(
                    sender=["Coder"],
                    tool_calls={"api_key": "must-not-leak"},
                ),
            ),
        )
    )

    app.run(timeout=10)

    assert not app.exception
    assert {"role": "System", "content": "Malformed tool call received."} in (
        app.session_state.messages
    )
    assert "must-not-leak" not in str(app.session_state.messages)


def test_display_is_defensive_against_preexisting_malformed_messages():
    app = AppTest.from_file(str(ROOT / "main.py")).run(timeout=10)
    app.session_state.messages = [
        {"role": ["Coder"], "content": {"api_key": "must-not-leak"}},
        object(),
    ]

    app.run(timeout=10)

    assert not app.exception
    rendered = [element.value for element in (*app.markdown, *app.text, *app.code)]
    assert "must-not-leak" not in str(rendered)


def test_unknown_event_becomes_a_system_message_without_dereferencing_content():
    app = AppTest.from_file(str(ROOT / "main.py")).run(timeout=10)
    runtime = FakeRuntime()
    app.session_state.runtime = runtime
    app.session_state.events = iter((SimpleNamespace(type="future_event"),))

    app.run(timeout=10)

    assert not app.exception
    assert {"role": "System", "content": "Unsupported event received."} in (
        app.session_state.messages
    )


def test_evidence_limit_failure_finishes_runtime_without_uncaught_exception():
    app = AppTest.from_file(str(ROOT / "main.py")).run(timeout=10)
    runtime = EvidenceLimitRuntime()
    app.session_state.runtime = runtime
    app.session_state.events = iter((SimpleNamespace(type="text", content="bounded"),))

    app.run(timeout=10)

    assert not app.exception
    assert runtime.finished == ["failed"]
    assert app.session_state.terminated is True
    assert runtime.failures == ["Event processing failed."]


def test_response_failure_is_recorded_before_failed_finalization():
    app = AppTest.from_file(str(ROOT / "main.py")).run(timeout=10)
    runtime = FakeRuntime()
    app.session_state.runtime = runtime
    app.session_state.events = iter((SimpleNamespace(type="input_request", content=None),))
    app.run(timeout=10)
    app.session_state.user_input = "answer"
    app.session_state.event = SimpleNamespace(
        content=SimpleNamespace(
            respond=lambda value: (_ for _ in ()).throw(RuntimeError("must-not-leak"))
        )
    )

    app.text_area(key="user_input").input("answer")
    app.button(key="submit_response").click().run(timeout=10)

    assert not app.exception
    assert runtime.failures == ["User response delivery failed."]
    assert runtime.finished == ["failed"]


def test_reset_closes_runtime_before_rerun():
    app = AppTest.from_file(str(ROOT / "main.py")).run(timeout=10)
    runtime = FakeRuntime()
    app.session_state.runtime = runtime

    app.button(key="restart").click().run(timeout=10)

    assert not app.exception
    assert runtime.closed == 1
    assert app.session_state.runtime is None


def test_reset_preserves_runtime_reference_when_cleanup_fails():
    app = AppTest.from_file(str(ROOT / "main.py")).run(timeout=10)
    runtime = CleanupFailRuntime()
    app.session_state.runtime = runtime

    app.button(key="restart").click().run(timeout=10)

    assert not app.exception
    assert runtime.closed == 1
    assert app.session_state.runtime is runtime
    assert all("must-not-leak" not in error.value for error in app.error)


def test_failed_restart_quarantines_queued_event_until_cleanup_retry_succeeds():
    app = AppTest.from_file(str(ROOT / "main.py")).run(timeout=10)
    runtime = RetryCleanupRuntime()
    queued_completion = SimpleNamespace(type="run_completion", content="must not run")
    app.session_state.runtime = runtime
    app.session_state.events = iter((queued_completion,))
    app.session_state.event = object()
    app.session_state.awaiting_response = True
    app.session_state.user_input = "stale response"
    app.session_state.last_agent_name = "Coder"

    app.button(key="restart").click().run(timeout=10)

    assert not app.exception
    assert app.session_state.runtime is runtime
    assert runtime.events == []
    assert runtime.finished == []
    assert app.session_state.terminated is True
    assert app.session_state.awaiting_response is False
    assert app.session_state.user_input == ""

    app.button(key="restart").click().run(timeout=10)

    assert not app.exception
    assert runtime.closed == 2
    assert runtime.events == []
    assert runtime.finished == []
    assert app.session_state.runtime is None


def test_completion_preserves_runtime_reference_when_finalization_fails():
    app = AppTest.from_file(str(ROOT / "main.py")).run(timeout=10)
    runtime = FinishFailRuntime()
    app.session_state.runtime = runtime
    app.session_state.events = iter(
        (SimpleNamespace(type="run_completion", content="Analysis complete."),)
    )

    app.run(timeout=10)

    assert not app.exception
    assert app.session_state.runtime is runtime
    assert all("must-not-leak" not in str(message) for message in app.session_state.messages)


def test_workspace_creation_failure_uses_generic_bounded_error(tmp_path, monkeypatch):
    secret_error = "api_key=must-not-leak " + "x" * 20_000
    monkeypatch.setattr(
        "agents.workspace.RunWorkspace.create",
        lambda: (_ for _ in ()).throw(RuntimeError(secret_error)),
    )
    app = AppTest.from_file(str(ROOT / "main.py")).run(timeout=10)
    configure_analysis(app, tmp_path)

    app.button(key="run_analysis").click().run(timeout=10)

    assert not app.exception
    assert app.error
    assert all("must-not-leak" not in error.value for error in app.error)
    assert max(len(error.value) for error in app.error) <= 200


def test_upload_failure_is_not_masked_or_leaked_when_workspace_close_fails(tmp_path, monkeypatch):
    class FailingWorkspace:
        def save_upload(self, name, data):
            raise ValueError("primary api_key=must-not-leak")

        def close(self):
            raise OSError("cleanup password=also-must-not-leak")

    monkeypatch.setattr("agents.workspace.RunWorkspace.create", FailingWorkspace)
    app = AppTest.from_file(str(ROOT / "main.py")).run(timeout=10)
    configure_analysis(app, tmp_path)

    app.button(key="run_analysis").click().run(timeout=10)

    assert not app.exception
    assert app.error
    assert all("must-not-leak" not in error.value for error in app.error)
