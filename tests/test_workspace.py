import json
import shutil

import pytest

from multi_agents.workspace import (
    RunWorkspace,
    UploadValidationError,
    WorkspaceLimitError,
    WorkspaceLimits,
)


CSV = b"customer_id,churn\n1,0\n2,1\n"


def test_save_upload_rejects_unsafe_names(tmp_path):
    workspace = RunWorkspace.create(tmp_path / "temp", tmp_path / "artifacts")
    with pytest.raises(UploadValidationError, match="safe base filename"):
        workspace.save_upload("../main.py", CSV)


def test_save_upload_records_hash_and_shape(tmp_path):
    workspace = RunWorkspace.create(tmp_path / "temp", tmp_path / "artifacts")
    path = workspace.save_upload("churn.csv", CSV)
    assert path.relative_to(workspace.root).as_posix() == "datasets/churn.csv"
    assert workspace.dataset_relative_paths == ("datasets/churn.csv",)
    manifest = workspace.dataset_manifest[0]
    assert manifest["name"] == "churn.csv"
    assert manifest["rows"] == 2
    assert manifest["columns"] == 2
    assert len(manifest["sha256"]) == 64


def test_save_upload_enforces_size_limit(tmp_path, monkeypatch):
    monkeypatch.setattr("multi_agents.workspace.MAX_UPLOAD_BYTES", 3)
    workspace = RunWorkspace.create(tmp_path / "temp", tmp_path / "artifacts")
    with pytest.raises(UploadValidationError, match="50 MiB"):
        workspace.save_upload("large.csv", CSV)


def test_save_upload_rejects_duplicate_names(tmp_path):
    workspace = RunWorkspace.create(tmp_path / "temp", tmp_path / "artifacts")
    workspace.save_upload("churn.csv", CSV)
    with pytest.raises(UploadValidationError, match="already uploaded"):
        workspace.save_upload("churn.csv", CSV)


def test_save_upload_rejects_invalid_csv(tmp_path):
    workspace = RunWorkspace.create(tmp_path / "temp", tmp_path / "artifacts")
    with pytest.raises(UploadValidationError, match="Invalid CSV"):
        workspace.save_upload("broken.csv", b'"unterminated')


@pytest.mark.parametrize(
    "step_name",
    ("../outside", "/tmp/outside", "nested/output", r"..\outside", ".."),
)
def test_record_output_rejects_unsafe_step_names(tmp_path, step_name):
    workspace = RunWorkspace.create(tmp_path / "temp", tmp_path / "artifacts")

    with pytest.raises(ValueError, match="safe filename stem"):
        workspace.record_output(step_name, "untrusted output")

    assert not any(workspace.output_dir.iterdir())


def test_record_output_rejects_collisions(tmp_path):
    workspace = RunWorkspace.create(tmp_path / "temp", tmp_path / "artifacts")
    workspace.record_output("step-001", "first")

    with pytest.raises(ValueError, match="already exists"):
        workspace.record_output("step-001", "second")

    assert (workspace.output_dir / "step-001.txt").read_text() == "first"


def test_finalize_retains_evidence_but_removes_uploaded_data(tmp_path):
    workspace = RunWorkspace.create(tmp_path / "temp", tmp_path / "artifacts")
    workspace.save_upload("churn.csv", CSV)
    workspace.record_prompt("Identify the drivers of churn.")
    code_path = workspace.record_code("print('ok')")
    workspace.record_output(code_path.stem, "ok\n")
    workspace.append_trace({"type": "text", "sender": "Coder"})
    workspace.record_recommendation("Prioritize customers with short tenure.")

    retained = workspace.finalize("completed", 1.25, ["one recovered failure"])

    assert not workspace.root.exists()
    assert (retained / "code" / "step-001.py").read_text() == "print('ok')\n"
    assert (retained / "outputs" / "step-001.txt").read_text() == "ok\n"
    assert (retained / "evidence" / "prompt.txt").read_text() == ("Identify the drivers of churn.")
    assert (retained / "evidence" / "recommendation.txt").read_text() == (
        "Prioritize customers with short tenure."
    )
    trace = [
        json.loads(line)
        for line in (retained / "evidence" / "trace.jsonl").read_text().splitlines()
    ]
    assert trace == [{"type": "text", "sender": "Coder"}]
    assert not (retained / "datasets").exists()
    run = json.loads((retained / "evidence" / "run.json").read_text())
    assert run["status"] == "completed"
    assert run["runtime_seconds"] == 1.25
    assert run["failures"] == ["one recovered failure"]
    assert run["dataset_manifest"][0]["name"] == "churn.csv"


def test_failed_run_can_finalize_without_recommendation(tmp_path):
    workspace = RunWorkspace.create(tmp_path / "temp", tmp_path / "artifacts")
    workspace.record_prompt("Analyze churn.")

    retained = workspace.finalize("failed", 0.5, ["analysis failed"])

    assert not (retained / "evidence" / "recommendation.txt").exists()


def test_finalize_copy_failure_preserves_workspace_and_cleans_staging(
    tmp_path,
    monkeypatch,
):
    artifacts_root = tmp_path / "artifacts"
    workspace = RunWorkspace.create(tmp_path / "temp", artifacts_root)
    workspace.record_code("print('retry')")
    original_copytree = shutil.copytree
    copy_calls = 0

    def fail_second_copy(source, destination):
        nonlocal copy_calls
        copy_calls += 1
        if copy_calls == 2:
            raise OSError("injected copy failure")
        return original_copytree(source, destination)

    monkeypatch.setattr("multi_agents.workspace.shutil.copytree", fail_second_copy)

    with pytest.raises(OSError, match="injected copy failure"):
        workspace.finalize("completed", 1.0, [])

    assert workspace.root.exists()
    assert (workspace.code_dir / "step-001.py").read_text() == "print('retry')\n"
    assert list(artifacts_root.iterdir()) == []

    monkeypatch.setattr("multi_agents.workspace.shutil.copytree", original_copytree)
    retained = workspace.finalize("completed", 1.0, [])
    assert (retained / "code" / "step-001.py").exists()


def test_artifact_record_limit_is_checked_before_writing(tmp_path):
    limits = WorkspaceLimits(max_record_bytes=4)
    workspace = RunWorkspace.create(
        tmp_path / "temp",
        tmp_path / "artifacts",
        limits=limits,
    )

    assert workspace.record_code("abc").read_bytes() == b"abc\n"
    with pytest.raises(WorkspaceLimitError, match="record limit"):
        workspace.record_output("step-001", "12345")
    with pytest.raises(WorkspaceLimitError, match="record limit"):
        workspace.append_trace({"message": "too large"})
    with pytest.raises(WorkspaceLimitError, match="record limit"):
        workspace.record_prompt("12345")
    with pytest.raises(WorkspaceLimitError, match="record limit"):
        workspace.record_recommendation("12345")

    assert not (workspace.output_dir / "step-001.txt").exists()
    assert not (workspace.evidence_dir / "trace.jsonl").exists()
    assert not (workspace.evidence_dir / "prompt.txt").exists()
    assert not (workspace.evidence_dir / "recommendation.txt").exists()

    second_workspace = RunWorkspace.create(
        tmp_path / "temp",
        tmp_path / "other-artifacts",
        limits=limits,
    )
    with pytest.raises(WorkspaceLimitError, match="record limit"):
        second_workspace.record_code("abcd")
    assert not any(second_workspace.code_dir.iterdir())


def test_artifact_run_limit_is_checked_before_writing(tmp_path):
    limits = WorkspaceLimits(max_record_bytes=100, max_run_bytes=8)
    workspace = RunWorkspace.create(
        tmp_path / "temp",
        tmp_path / "artifacts",
        limits=limits,
    )
    workspace.record_code("abc")
    workspace.record_output("step-001", "1234")

    with pytest.raises(WorkspaceLimitError, match="run limit"):
        workspace.append_trace({"message": "over aggregate limit"})

    assert not (workspace.evidence_dir / "trace.jsonl").exists()


def test_artifact_count_limits_are_checked_before_writing(tmp_path):
    limits = WorkspaceLimits(
        max_code_records=1,
        max_output_records=1,
        max_trace_records=1,
    )
    workspace = RunWorkspace.create(
        tmp_path / "temp",
        tmp_path / "artifacts",
        limits=limits,
    )
    workspace.record_code("first")
    workspace.record_output("first", "first")
    workspace.append_trace({"sequence": 1})

    with pytest.raises(WorkspaceLimitError, match="code count limit"):
        workspace.record_code("second")
    with pytest.raises(WorkspaceLimitError, match="output count limit"):
        workspace.record_output("second", "second")
    with pytest.raises(WorkspaceLimitError, match="trace count limit"):
        workspace.append_trace({"sequence": 2})

    assert [path.name for path in workspace.code_dir.iterdir()] == ["step-001.py"]
    assert [path.name for path in workspace.output_dir.iterdir()] == ["first.txt"]
    assert len((workspace.evidence_dir / "trace.jsonl").read_text().splitlines()) == 1


def test_close_is_idempotent(tmp_path):
    workspace = RunWorkspace.create(tmp_path / "temp", tmp_path / "artifacts")
    root = workspace.root

    workspace.close()
    workspace.close()

    assert not root.exists()
