import hashlib
import json
import os
from datetime import datetime, timedelta

import pytest
from autogen.agentchat.group import ContextVariables
from autogen.coding import DockerCommandLineCodeExecutor

from multi_agents.execution import CodeRunner, create_docker_executor
from multi_agents.workspace import RunWorkspace


EXECUTED_CODE = """\
import fcntl
import os
import socket
import struct
from pathlib import Path

api_key_absent = os.getenv("OPENAI_API_KEY") is None
project_source_absent = not Path("/project/main.py").exists()

configured_interfaces = set()
with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as interface_socket:
    for _, name in socket.if_nameindex():
        request = struct.pack("256s", name.encode()[:15])
        try:
            fcntl.ioctl(interface_socket.fileno(), 0x8915, request)
        except OSError:
            continue
        configured_interfaces.add(name)
for line in Path("/proc/net/if_inet6").read_text().splitlines():
    configured_interfaces.add(line.split()[-1])

ipv4_routes = [
    line.split()
    for line in Path("/proc/net/route").read_text().splitlines()[1:]
    if line.split() and int(line.split()[3], 16) & 1
]
ipv6_routes = [
    line.split()
    for line in Path("/proc/net/ipv6_route").read_text().splitlines()
    if line.split() and int(line.split()[8], 16) & 1
]
default_route_count = sum(route[1] == "00000000" for route in ipv4_routes)
default_route_count += sum(
    route[0] == "0" * 32 and route[1] == "00" for route in ipv6_routes
)
non_loopback_route_count = sum(route[0] != "lo" for route in ipv4_routes)
non_loopback_route_count += sum(route[9] != "lo" for route in ipv6_routes)

root_statvfs_read_only = bool(os.statvfs("/").f_flag & os.ST_RDONLY)
root_mount = next(
    line.split()
    for line in Path("/proc/self/mountinfo").read_text().splitlines()
    if line.split()[4] == "/"
)
root_mount_read_only = "ro" in root_mount[5].split(",")

assert configured_interfaces == {"lo"}
assert default_route_count == 0
assert non_loopback_route_count == 0
assert root_statvfs_read_only
assert root_mount_read_only

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
print(f"network_interfaces={','.join(sorted(configured_interfaces))}")
print(f"default_route_count={default_route_count}")
print(f"non_loopback_route_count={non_loopback_route_count}")
print(f"root_statvfs_read_only={root_statvfs_read_only}")
print(f"root_mount_read_only={root_mount_read_only}")
print("python_execution=ok")
"""
EXPECTED_OUTPUT = """\
api_key_absent=True
project_source_absent=True
network_interfaces=lo
default_route_count=0
non_loopback_route_count=0
root_statvfs_read_only=True
root_mount_read_only=True
python_execution=ok
"""
UPLOAD = b"customer_id,churn\n1,0\n"
EXPECTED_MANIFEST = [
    {
        "name": "customers.csv",
        "sha256": hashlib.sha256(UPLOAD).hexdigest(),
        "rows": 1,
        "columns": 2,
    }
]


def _stop_executor_and_close_workspace(executor, workspace):
    stop_error = None
    try:
        if executor is not None:
            executor.stop()
    except BaseException as exc:
        stop_error = exc
    try:
        workspace.close()
    except BaseException as close_error:
        if stop_error is not None:
            raise stop_error from close_error
        raise
    if stop_error is not None:
        raise stop_error


def test_cleanup_closes_workspace_when_executor_stop_fails(tmp_path):
    class StopFailureExecutor:
        def __init__(self):
            self.stop_calls = 0

        def stop(self):
            self.stop_calls += 1
            raise RuntimeError("injected stop failure")

    workspace = RunWorkspace.create(tmp_path / "temp", tmp_path / "artifacts")
    root = workspace.root
    private_root = workspace.code_dir.parent
    executor = StopFailureExecutor()

    with pytest.raises(RuntimeError, match="injected stop failure"):
        _stop_executor_and_close_workspace(executor, workspace)

    assert executor.stop_calls == 1
    assert not root.exists()
    assert not private_root.exists()


def test_cleanup_closes_workspace_when_executor_was_not_created(tmp_path):
    workspace = RunWorkspace.create(tmp_path / "temp", tmp_path / "artifacts")
    root = workspace.root
    private_root = workspace.code_dir.parent

    _stop_executor_and_close_workspace(None, workspace)

    assert not root.exists()
    assert not private_root.exists()


@pytest.mark.docker
@pytest.mark.skipif(
    os.environ.get("RUN_DOCKER_TESTS") != "1",
    reason="opt-in Docker test",
)
def test_executor_isolated_and_evidence_survives(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-enter-container")
    workspace = RunWorkspace.create(tmp_path / "temp", tmp_path / "artifacts")
    executor = None
    cleanup_executor = None
    try:
        upload = workspace.save_upload("customers.csv", UPLOAD)
        executor = create_docker_executor(workspace)
        cleanup_executor = executor
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
        mounted_scratch_paths = (
            upload,
            workspace.root / "scratch-only.txt",
            workspace.root / "code" / "step-001.py",
            workspace.root / "outputs" / "step-001.txt",
            workspace.root / "evidence" / "run.json",
        )

        executor.stop()
        cleanup_executor = None
        retained = workspace.finalize(
            "completed",
            12.5,
            ["recovered executor warning"],
        )

        assert retained == tmp_path / "artifacts" / workspace.run_id
        assert (retained / "code" / "step-001.py").read_text() == EXECUTED_CODE
        assert (retained / "outputs" / "step-001.txt").read_text() == EXPECTED_OUTPUT
        assert not (retained / "code" / "step-001.py").is_symlink()
        assert not (retained / "outputs" / "step-001.txt").is_symlink()
        retained_directories = {
            path.relative_to(retained).as_posix()
            for path in retained.rglob("*")
            if path.is_dir() and not path.is_symlink()
        }
        retained_files = {
            path.relative_to(retained).as_posix()
            for path in retained.rglob("*")
            if path.is_file() or path.is_symlink()
        }
        assert retained_directories == {"code", "outputs", "evidence"}
        assert retained_files == {
            "code/step-001.py",
            "outputs/step-001.txt",
            "evidence/run.json",
        }
        assert all(
            not path.exists() and not path.is_symlink()
            for path in mounted_scratch_paths
        )
        run_evidence = json.loads((retained / "evidence" / "run.json").read_text())
        finished_at_text = run_evidence.pop("finished_at")
        finished_at = datetime.fromisoformat(finished_at_text)
        assert finished_at_text.endswith("+00:00")
        assert finished_at.utcoffset() == timedelta(0)
        assert run_evidence == {
            "run_id": workspace.run_id,
            "status": "completed",
            "runtime_seconds": 12.5,
            "failures": ["recovered executor warning"],
            "dataset_manifest": EXPECTED_MANIFEST,
        }
        assert not workspace.root.exists()
    finally:
        _stop_executor_and_close_workspace(cleanup_executor, workspace)
