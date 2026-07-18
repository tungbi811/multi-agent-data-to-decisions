from __future__ import annotations

import hashlib
import io
import json
import shutil
import tempfile
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd


MAX_UPLOAD_BYTES = 50 * 1024 * 1024


class UploadValidationError(ValueError):
    pass


class RunWorkspace:
    def __init__(self, root: Path, artifacts_root: Path, run_id: str) -> None:
        self.root = root
        self.artifacts_root = artifacts_root
        self.run_id = run_id
        self.dataset_dir = root / "datasets"
        self.code_dir = root / "code"
        self.output_dir = root / "outputs"
        self.evidence_dir = root / "evidence"
        for directory in (self.dataset_dir, self.code_dir, self.output_dir, self.evidence_dir):
            directory.mkdir(parents=True, exist_ok=True)
        self._datasets: list[dict[str, Any]] = []
        self._code_counter = 0
        self._trace_path = self.evidence_dir / "trace.jsonl"

    @classmethod
    def create(
        cls,
        temp_parent: Path | None = None,
        artifacts_root: Path | None = None,
    ) -> "RunWorkspace":
        parent = None if temp_parent is None else str(temp_parent)
        if temp_parent is not None:
            temp_parent.mkdir(parents=True, exist_ok=True)
        root = Path(tempfile.mkdtemp(prefix="auto-ds-", dir=parent))
        return cls(root, artifacts_root or Path("artifacts/runs"), uuid.uuid4().hex)

    @property
    def dataset_relative_paths(self) -> tuple[str, ...]:
        return tuple(f"datasets/{item['name']}" for item in self._datasets)

    @property
    def dataset_manifest(self) -> tuple[dict[str, Any], ...]:
        return tuple(dict(item) for item in self._datasets)

    def save_upload(self, name: str, data: bytes) -> Path:
        filename = Path(name)
        if filename.name != name or name in {"", ".", ".."}:
            raise UploadValidationError("Upload must use a safe base filename.")
        if filename.suffix.lower() != ".csv":
            raise UploadValidationError("Only CSV uploads are supported.")
        if len(data) > MAX_UPLOAD_BYTES:
            raise UploadValidationError("CSV uploads must not exceed 50 MiB.")
        try:
            frame = pd.read_csv(io.BytesIO(data))
        except Exception as exc:
            raise UploadValidationError(f"Invalid CSV: {exc}") from exc
        destination = self.dataset_dir / filename.name
        if destination.exists():
            raise UploadValidationError(f"{filename.name!r} was already uploaded.")
        destination.write_bytes(data)
        self._datasets.append(
            {
                "name": filename.name,
                "sha256": hashlib.sha256(data).hexdigest(),
                "rows": int(frame.shape[0]),
                "columns": int(frame.shape[1]),
            }
        )
        return destination

    def record_code(self, code: str) -> Path:
        self._code_counter += 1
        path = self.code_dir / f"step-{self._code_counter:03d}.py"
        path.write_text(code.rstrip() + "\n", encoding="utf-8")
        return path

    def record_output(self, step_name: str, output: str) -> Path:
        path = self.output_dir / f"{step_name}.txt"
        path.write_text(output, encoding="utf-8")
        return path

    def append_trace(self, event: dict[str, Any]) -> None:
        with self._trace_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event, default=str) + "\n")

    def finalize(self, status: str, runtime_seconds: float, failures: list[str]) -> Path:
        run = {
            "run_id": self.run_id,
            "status": status,
            "runtime_seconds": runtime_seconds,
            "failures": failures,
            "dataset_manifest": self._datasets,
            "finished_at": datetime.now(UTC).isoformat(),
        }
        (self.evidence_dir / "run.json").write_text(
            json.dumps(run, indent=2),
            encoding="utf-8",
        )
        retained = self.artifacts_root / self.run_id
        retained.mkdir(parents=True, exist_ok=False)
        for name in ("code", "outputs", "evidence"):
            shutil.copytree(self.root / name, retained / name)
        shutil.rmtree(self.root)
        return retained

    def close(self) -> None:
        if self.root.exists():
            shutil.rmtree(self.root)
