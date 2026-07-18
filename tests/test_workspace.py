import json

import pytest

from multi_agents.workspace import RunWorkspace, UploadValidationError


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


def test_finalize_retains_evidence_but_removes_uploaded_data(tmp_path):
    workspace = RunWorkspace.create(tmp_path / "temp", tmp_path / "artifacts")
    workspace.save_upload("churn.csv", CSV)
    code_path = workspace.record_code("print('ok')")
    workspace.record_output(code_path.stem, "ok\n")
    workspace.append_trace({"type": "text", "sender": "Coder"})

    retained = workspace.finalize("completed", 1.25, [])

    assert not workspace.root.exists()
    assert (retained / "code" / "step-001.py").read_text() == "print('ok')\n"
    assert (retained / "outputs" / "step-001.txt").read_text() == "ok\n"
    assert not (retained / "datasets").exists()
    run = json.loads((retained / "evidence" / "run.json").read_text())
    assert run["status"] == "completed"
    assert run["dataset_manifest"][0]["name"] == "churn.csv"
