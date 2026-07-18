import json
import os

import pytest
from autogen.agentchat.group import ContextVariables
from autogen.coding import DockerCommandLineCodeExecutor

from multi_agents.execution import CodeRunner, create_docker_executor
from multi_agents.workspace import RunWorkspace


pytestmark = pytest.mark.docker

EXECUTED_CODE = """\
import os
import socket
from pathlib import Path

api_key_absent = os.getenv("OPENAI_API_KEY") is None
project_source_absent = not Path("/project/main.py").exists()
try:
    socket.create_connection(("1.1.1.1", 53), timeout=1)
except OSError:
    network_blocked = True
else:
    network_blocked = False
try:
    Path("/executor-root-write.txt").write_text("must fail")
except OSError:
    root_read_only = True
else:
    root_read_only = False

Path("scratch-only.txt").write_text("container scratch")
Path("code").mkdir()
Path("code/step-001.py").write_text("print('tampered')\\n")
Path("outputs").mkdir()
Path("outputs/step-001.txt").symlink_to("/tmp/container-redirect.txt")
Path("outputs/step-001.txt").write_text("redirected scratch")
Path("evidence").mkdir()
Path("evidence/run.json").write_text('{"status": "tampered"}')

print(f"api_key_absent={api_key_absent}")
print(f"project_source_absent={project_source_absent}")
print(f"network_blocked={network_blocked}")
print(f"root_read_only={root_read_only}")
print("python_execution=ok")
"""
EXPECTED_OUTPUT = """\
api_key_absent=True
project_source_absent=True
network_blocked=True
root_read_only=True
python_execution=ok
"""
UPLOAD = b"customer_id,churn\n1,0\n"


@pytest.mark.skipif(
    os.environ.get("RUN_DOCKER_TESTS") != "1",
    reason="opt-in Docker test",
)
def test_executor_isolated_and_evidence_survives(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-enter-container")
    workspace = RunWorkspace.create(tmp_path / "temp", tmp_path / "artifacts")
    executor = None
    stopped = False
    try:
        upload = workspace.save_upload("customers.csv", UPLOAD)
        executor = create_docker_executor(workspace)
        assert isinstance(executor, DockerCommandLineCodeExecutor)
        assert executor.execution_policies["python"] is True
        assert all(
            not allowed
            for language, allowed in executor.execution_policies.items()
            if language != "python"
        )

        reply = CodeRunner(workspace, executor).run_code(
            EXECUTED_CODE,
            ContextVariables(data={"current_agent": "DataScientist"}),
        )
        assert reply.message == f"Output:\n{EXPECTED_OUTPUT}"
        assert (workspace.code_dir / "step-001.py").read_text() == EXECUTED_CODE
        assert (workspace.output_dir / "step-001.txt").read_text() == EXPECTED_OUTPUT
        assert upload.exists()
        assert (workspace.root / "scratch-only.txt").read_text() == "container scratch"
        assert (workspace.root / "code" / "step-001.py").read_text() == (
            "print('tampered')\n"
        )

        executor.stop()
        stopped = True
        retained = workspace.finalize("completed", 0.0, [])

        assert retained == tmp_path / "artifacts" / workspace.run_id
        assert (retained / "code" / "step-001.py").read_text() == EXECUTED_CODE
        assert (retained / "outputs" / "step-001.txt").read_text() == EXPECTED_OUTPUT
        assert not (retained / "datasets").exists()
        assert not (retained / "scratch-only.txt").exists()
        run_evidence = json.loads((retained / "evidence" / "run.json").read_text())
        assert run_evidence["run_id"] == workspace.run_id
        assert run_evidence["status"] == "completed"
        assert run_evidence["dataset_manifest"][0]["name"] == "customers.csv"
        assert not workspace.root.exists()
    finally:
        if executor is not None and not stopped:
            executor.stop()
        workspace.close()
