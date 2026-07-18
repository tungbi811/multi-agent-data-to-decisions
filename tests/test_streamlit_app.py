import ast
from pathlib import Path
from types import SimpleNamespace

from streamlit.testing.v1 import AppTest


ROOT = Path(__file__).parents[1]


class FakeRuntime:
    def __init__(self):
        self.events = []
        self.finished = []
        self.closed = 0

    def record_event(self, event):
        self.events.append(event)

    def finish(self, status):
        self.finished.append(status)

    def close(self):
        self.closed += 1


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
    assert app.title[0].value == "🤖 Multi-Agent for Data Science"


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
        "multi_agents.runtime.AnalysisRuntime.start",
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


def test_iterator_failure_finishes_runtime_without_uncaught_exception():
    app = AppTest.from_file(str(ROOT / "main.py")).run(timeout=10)
    runtime = FakeRuntime()
    app.session_state.runtime = runtime
    app.session_state.events = FailingIterator()

    app.run(timeout=10)

    assert not app.exception
    assert runtime.finished == ["failed"]
    assert app.session_state.terminated is True


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


def test_reset_closes_runtime_before_rerun():
    app = AppTest.from_file(str(ROOT / "main.py")).run(timeout=10)
    runtime = FakeRuntime()
    app.session_state.runtime = runtime

    app.button(key="restart").click().run(timeout=10)

    assert not app.exception
    assert runtime.closed == 1
    assert app.session_state.runtime is None
