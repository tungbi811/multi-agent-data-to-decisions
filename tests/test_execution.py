from types import SimpleNamespace

from autogen.agentchat.group import ContextVariables

from multi_agents.execution import CodeRunner, create_docker_executor
from multi_agents.workspace import RunWorkspace


class FakeExecutor:
    def __init__(self, outcomes):
        self.outcomes = iter(outcomes)
        self.calls = 0
        self.restarts = 0
        self.stops = 0

    def execute_code_blocks(self, blocks):
        self.calls += 1
        outcome = next(self.outcomes)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    def restart(self):
        self.restarts += 1

    def stop(self):
        self.stops += 1


def context():
    return ContextVariables(data={"current_agent": "DataScientist"})


def test_success_records_code_and_output(tmp_path):
    workspace = RunWorkspace.create(tmp_path / "temp", tmp_path / "artifacts")
    executor = FakeExecutor([SimpleNamespace(exit_code=0, output="42\n")])
    result = CodeRunner(workspace, executor).run_code("print(42)", context())
    assert executor.calls == 1
    assert (workspace.code_dir / "step-001.py").read_text() == "print(42)\n"
    assert (workspace.output_dir / "step-001.txt").read_text() == "42\n"
    assert "Output:\n42" in result.message


def test_exception_restarts_once_then_succeeds(tmp_path):
    workspace = RunWorkspace.create(tmp_path / "temp", tmp_path / "artifacts")
    executor = FakeExecutor([RuntimeError("first"), SimpleNamespace(exit_code=0, output="ok")])
    result = CodeRunner(workspace, executor).run_code("print('ok')", context())
    assert executor.calls == 2
    assert executor.restarts == 1
    assert "Output:\nok" in result.message


def test_second_exception_returns_controlled_failure(tmp_path):
    workspace = RunWorkspace.create(tmp_path / "temp", tmp_path / "artifacts")
    executor = FakeExecutor([RuntimeError("first"), RuntimeError("second")])
    result = CodeRunner(workspace, executor).run_code("raise RuntimeError", context())
    assert executor.calls == 2
    assert executor.restarts == 1
    assert "Execution failed after retry: second" in result.message


def test_nonzero_result_preserves_end_of_error(tmp_path):
    workspace = RunWorkspace.create(tmp_path / "temp", tmp_path / "artifacts")
    output = "header\ncontext\nValueError: useful ending"
    executor = FakeExecutor([SimpleNamespace(exit_code=1, output=output)])
    result = CodeRunner(workspace, executor).run_code("bad()", context())
    assert "ValueError: useful ending" in result.message


def test_docker_factory_applies_isolation_limits(tmp_path, monkeypatch):
    captured = {}

    def fake_executor(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace()

    monkeypatch.setattr("multi_agents.execution.DockerCommandLineCodeExecutor", fake_executor)
    workspace = RunWorkspace.create(tmp_path / "temp", tmp_path / "artifacts")
    create_docker_executor(workspace)
    options = captured["container_create_kwargs"]
    assert options["network_disabled"] is True
    assert options["mem_limit"] == "2g"
    assert options["nano_cpus"] == 2_000_000_000
    assert options["pids_limit"] == 256
    assert "OPENAI_API_KEY" not in options["environment"]
