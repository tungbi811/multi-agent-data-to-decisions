import os
from types import SimpleNamespace

from autogen.agentchat.group import ContextVariables
from autogen.coding import DockerCommandLineCodeExecutor

from multi_agents.execution import MAX_OUTPUT_CHARS, CodeRunner, create_docker_executor
from multi_agents.workspace import RunWorkspace


class FakeExecutor:
    def __init__(self, outcomes, restart_error=None):
        self.outcomes = iter(outcomes)
        self.restart_error = restart_error
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
        if self.restart_error is not None:
            raise self.restart_error

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
    evidence = (workspace.output_dir / "step-001.txt").read_text()
    assert "Execution attempt 1 failed:\nfirst" in evidence
    assert "Executor restart succeeded." in evidence
    assert "Execution attempt 2 output:\nok" in evidence


def test_second_exception_returns_controlled_failure(tmp_path):
    workspace = RunWorkspace.create(tmp_path / "temp", tmp_path / "artifacts")
    executor = FakeExecutor([RuntimeError("first"), RuntimeError("second")])
    result = CodeRunner(workspace, executor).run_code("raise RuntimeError", context())
    assert executor.calls == 2
    assert executor.restarts == 1
    assert "Execution failed after retry: second" in result.message
    evidence = (workspace.output_dir / "step-001.txt").read_text()
    assert "Execution attempt 1 failed:\nfirst" in evidence
    assert "Executor restart succeeded." in evidence
    assert "Execution attempt 2 failed:\nsecond" in evidence


def test_restart_exception_returns_controlled_failure_with_evidence(tmp_path):
    workspace = RunWorkspace.create(tmp_path / "temp", tmp_path / "artifacts")
    executor = FakeExecutor(
        [RuntimeError("execute failed")], restart_error=RuntimeError("restart failed")
    )

    result = CodeRunner(workspace, executor).run_code("print('ok')", context())

    assert executor.calls == 1
    assert executor.restarts == 1
    assert "Execution failed during restart: restart failed" in result.message
    evidence = (workspace.output_dir / "step-001.txt").read_text()
    assert "Execution attempt 1 failed:\nexecute failed" in evidence
    assert "Executor restart failed:\nrestart failed" in evidence


def test_oversized_retry_exception_is_bounded_for_agent_but_preserved_in_evidence(
    tmp_path,
):
    workspace = RunWorkspace.create(tmp_path / "temp", tmp_path / "artifacts")
    oversized = "x" * (MAX_OUTPUT_CHARS + 100) + "useful ending"
    executor = FakeExecutor([RuntimeError("first"), RuntimeError(oversized)])

    result = CodeRunner(workspace, executor).run_code("bad()", context())

    assert len(result.message) <= MAX_OUTPUT_CHARS
    assert result.message.endswith("useful ending")
    evidence = (workspace.output_dir / "step-001.txt").read_text()
    assert "Execution attempt 1 failed:\nfirst" in evidence
    assert oversized in evidence


def test_nonzero_result_preserves_end_of_error(tmp_path):
    workspace = RunWorkspace.create(tmp_path / "temp", tmp_path / "artifacts")
    output = "header\ncontext\nValueError: useful ending"
    executor = FakeExecutor([SimpleNamespace(exit_code=1, output=output)])
    result = CodeRunner(workspace, executor).run_code("bad()", context())
    assert "ValueError: useful ending" in result.message


def test_stop_delegates_to_executor(tmp_path):
    workspace = RunWorkspace.create(tmp_path / "temp", tmp_path / "artifacts")
    executor = FakeExecutor([])

    CodeRunner(workspace, executor).stop()

    assert executor.stops == 1


def test_docker_factory_applies_python_only_isolation(tmp_path, monkeypatch):
    captured = {}
    default_policy = DockerCommandLineCodeExecutor.DEFAULT_EXECUTION_POLICY.copy()

    class FakeDockerExecutor:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.execution_policies = default_policy.copy()
            self.execution_policies.update(kwargs["execution_policies"])

    monkeypatch.setattr("multi_agents.execution.DockerCommandLineCodeExecutor", FakeDockerExecutor)
    workspace = RunWorkspace.create(tmp_path / "temp", tmp_path / "artifacts")
    executor = create_docker_executor(workspace)

    assert set(captured) == {
        "image",
        "timeout",
        "work_dir",
        "execution_policies",
        "container_create_kwargs",
    }
    assert captured["image"] == "auto-ds-executor:0.1"
    assert captured["timeout"] == 300
    assert captured["work_dir"] == workspace.root
    expected_policy = {language: language == "python" for language in default_policy}
    assert captured["execution_policies"] == expected_policy
    assert executor.execution_policies == expected_policy
    assert executor.execution_policies["pwsh"] is False
    assert executor.execution_policies["powershell"] is False
    assert executor.execution_policies["ps1"] is False

    options = captured["container_create_kwargs"]
    assert options == {
        "network_disabled": True,
        "mem_limit": "2g",
        "nano_cpus": 2_000_000_000,
        "pids_limit": 256,
        "read_only": True,
        "tmpfs": {"/tmp": "rw,nosuid,size=268435456"},
        "user": f"{os.getuid()}:{os.getgid()}",
        "environment": {
            "PYTHONDONTWRITEBYTECODE": "1",
            "MPLCONFIGDIR": "/tmp/matplotlib",
        },
    }
