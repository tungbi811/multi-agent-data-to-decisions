from __future__ import annotations

import hashlib
import io
import json
import re
import shutil
import tempfile
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd


MAX_UPLOAD_BYTES = 50 * 1024 * 1024
MAX_OUTPUT_STEM_CHARS = 100
SAFE_STEM = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*")


class UploadValidationError(ValueError):
    pass


class WorkspaceLimitError(ValueError):
    pass


@dataclass(frozen=True)
class WorkspaceLimits:
    max_record_bytes: int = 1024 * 1024
    max_run_bytes: int = 20 * 1024 * 1024
    max_code_records: int = 100
    max_output_records: int = 100
    max_trace_records: int = 1000

    def __post_init__(self) -> None:
        if (
            min(
                self.max_record_bytes,
                self.max_run_bytes,
                self.max_code_records,
                self.max_output_records,
                self.max_trace_records,
            )
            <= 0
        ):
            raise ValueError("Workspace limits must be positive integers.")


class RunWorkspace:
    def __init__(
        self,
        root: Path,
        artifacts_root: Path,
        run_id: str,
        limits: WorkspaceLimits,
    ) -> None:
        self.root = root
        self.artifacts_root = artifacts_root
        self.run_id = run_id
        self.limits = limits
        self.dataset_dir = root / "datasets"
        self.code_dir = root / "code"
        self.output_dir = root / "outputs"
        self.evidence_dir = root / "evidence"
        for directory in (self.dataset_dir, self.code_dir, self.output_dir, self.evidence_dir):
            directory.mkdir(parents=True, exist_ok=True)
        self._datasets: list[dict[str, Any]] = []
        self._code_counter = 0
        self._output_counter = 0
        self._trace_counter = 0
        self._artifact_bytes = 0
        self._trace_path = self.evidence_dir / "trace.jsonl"

    @classmethod
    def create(
        cls,
        temp_parent: Path | None = None,
        artifacts_root: Path | None = None,
        *,
        limits: WorkspaceLimits | None = None,
    ) -> "RunWorkspace":
        parent = None if temp_parent is None else str(temp_parent)
        if temp_parent is not None:
            temp_parent.mkdir(parents=True, exist_ok=True)
        root = Path(tempfile.mkdtemp(prefix="auto-ds-", dir=parent))
        return cls(
            root,
            artifacts_root or Path("artifacts/runs"),
            uuid.uuid4().hex,
            limits or WorkspaceLimits(),
        )

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
        self._check_count("code", self._code_counter, self.limits.max_code_records)
        payload = (code.rstrip() + "\n").encode()
        self._check_size("code", payload)
        path = self.code_dir / f"step-{self._code_counter + 1:03d}.py"
        path.write_bytes(payload)
        self._code_counter += 1
        self._artifact_bytes += len(payload)
        return path

    def record_output(self, step_name: str, output: str) -> Path:
        if len(step_name) > MAX_OUTPUT_STEM_CHARS or SAFE_STEM.fullmatch(step_name) is None:
            raise ValueError(
                "Output step name must be a safe filename stem of at most "
                f"{MAX_OUTPUT_STEM_CHARS} ASCII characters."
            )
        path = self.output_dir / f"{step_name}.txt"
        if path.exists():
            raise ValueError(f"Output for {step_name!r} already exists.")
        self._check_count(
            "output",
            self._output_counter,
            self.limits.max_output_records,
        )
        payload = output.encode()
        self._check_size("output", payload)
        path.write_bytes(payload)
        self._output_counter += 1
        self._artifact_bytes += len(payload)
        return path

    def record_prompt(self, prompt: str) -> Path:
        return self._record_text_evidence("prompt.txt", prompt)

    def record_recommendation(self, recommendation: str) -> Path:
        return self._record_text_evidence("recommendation.txt", recommendation)

    def _record_text_evidence(self, name: str, text: str) -> Path:
        path = self.evidence_dir / name
        if path.exists():
            raise ValueError(f"Evidence file {name!r} already exists.")
        payload = text.encode()
        self._check_size(name, payload)
        path.write_bytes(payload)
        self._artifact_bytes += len(payload)
        return path

    def append_trace(self, event: dict[str, Any]) -> None:
        self._check_count(
            "trace",
            self._trace_counter,
            self.limits.max_trace_records,
        )
        payload = (json.dumps(event, default=str) + "\n").encode()
        self._check_size("trace", payload)
        with self._trace_path.open("ab") as stream:
            stream.write(payload)
        self._trace_counter += 1
        self._artifact_bytes += len(payload)

    def _check_count(self, kind: str, count: int, limit: int) -> None:
        if count >= limit:
            raise WorkspaceLimitError(f"Workspace {kind} count limit exceeded.")

    def _check_size(self, kind: str, payload: bytes) -> None:
        if len(payload) > self.limits.max_record_bytes:
            raise WorkspaceLimitError(f"Workspace {kind} record limit exceeded.")
        if self._artifact_bytes + len(payload) > self.limits.max_run_bytes:
            raise WorkspaceLimitError(f"Workspace {kind} run limit exceeded.")

    def finalize(self, status: str, runtime_seconds: float, failures: list[str]) -> Path:
        run = {
            "run_id": self.run_id,
            "status": status,
            "runtime_seconds": runtime_seconds,
            "failures": failures,
            "dataset_manifest": self._datasets,
            "finished_at": datetime.now(UTC).isoformat(),
        }
        run_payload = json.dumps(run, indent=2).encode()
        self._check_size("run metadata", run_payload)
        (self.evidence_dir / "run.json").write_bytes(run_payload)
        retained = self.artifacts_root / self.run_id
        self.artifacts_root.mkdir(parents=True, exist_ok=True)
        if retained.exists():
            raise FileExistsError(f"Retained run {self.run_id!r} already exists.")
        staging = Path(tempfile.mkdtemp(prefix=f".{self.run_id}-", dir=self.artifacts_root))
        try:
            for name in ("code", "outputs", "evidence"):
                shutil.copytree(self.root / name, staging / name)
            staging.rename(retained)
        finally:
            if staging.exists():
                shutil.rmtree(staging)
        shutil.rmtree(self.root)
        return retained

    def close(self) -> None:
        if self.root.exists():
            shutil.rmtree(self.root)
