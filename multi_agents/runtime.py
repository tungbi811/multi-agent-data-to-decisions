from __future__ import annotations

import contextlib
import re
import time
from typing import Any

from .execution import CodeRunner, create_docker_executor
from .group_chat import GroupChat
from .workspace import RunWorkspace
from utils.events import safe_content, safe_event_type, safe_role


MAX_FAILURE_CHARS = 1000
MAX_FAILURES = 20
SECRET_ASSIGNMENT = re.compile(r"(?i)\b(api[_ -]?key|token|secret|password)\b\s*[:=]\s*[^\s;,]+")
OPENAI_KEY = re.compile(r"\bsk-[A-Za-z0-9_-]+\b")


class AnalysisRuntime:
    def __init__(
        self,
        workspace: RunWorkspace,
        code_runner: CodeRunner,
        group_chat: GroupChat,
    ) -> None:
        self.workspace = workspace
        self.started_at = time.monotonic()
        self.failures: list[str] = []
        self.code_runner = code_runner
        self.group_chat = group_chat
        self.finished = False
        self._runner_stop_attempted = False
        self._workspace_cleaned = False
        self._pending_status: str | None = None
        self._pending_elapsed: float | None = None
        self._run_started = False
        self._recommendation_recorded = False

    @classmethod
    def start(cls, api_key: str, workspace: RunWorkspace) -> AnalysisRuntime:
        executor = None
        code_runner = None
        try:
            executor = create_docker_executor(workspace)
            code_runner = CodeRunner(workspace, executor)
            group_chat = GroupChat(api_key, workspace, code_runner)
        except Exception:
            cleanup_target = code_runner if code_runner is not None else executor
            if cleanup_target is not None:
                with contextlib.suppress(Exception):
                    cleanup_target.stop()
            with contextlib.suppress(Exception):
                workspace.close()
            raise
        return cls(workspace, code_runner, group_chat)

    def run(self, requirement: str):
        if self._run_started:
            raise RuntimeError("Analysis runtime can only be run once.")
        self.workspace.record_prompt(requirement)
        self._run_started = True
        return self.group_chat.run(self.workspace.dataset_relative_paths, requirement)

    def record_event(self, event: Any) -> None:
        content = getattr(event, "content", "")
        sender = safe_role(getattr(content, "sender", "System"))
        message = safe_content(
            getattr(content, "content", content if isinstance(content, str) else None)
        )
        self.workspace.append_trace(
            {
                "type": safe_event_type(event),
                "sender": sender,
                "content": message,
            }
        )
        if self._recommendation_recorded:
            return
        if getattr(content, "sender", None) != "BusinessTranslator":
            return
        recommendation = getattr(content, "content", None)
        if not isinstance(recommendation, str):
            return
        bounded = _bounded_text(
            _redact_secrets(recommendation),
            self.workspace.limits.max_record_bytes,
        )
        self.workspace.record_recommendation(bounded)
        self._recommendation_recorded = True

    def record_failure(self, reason: str) -> None:
        if len(self.failures) >= MAX_FAILURES:
            return
        message = _redact_secrets(reason if isinstance(reason, str) else "Runtime failure.")
        message = message[:MAX_FAILURE_CHARS]
        if message and message not in self.failures:
            self.failures.append(message)

    def finish(self, status: str) -> None:
        if self._workspace_cleaned:
            return
        if self._pending_status is None:
            self._pending_status = status
            self._pending_elapsed = time.monotonic() - self.started_at
        self._stop_runner()
        try:
            self.workspace.finalize(self._pending_status, self._pending_elapsed, self.failures)
        except Exception as exc:
            self.record_failure(f"Workspace finalization failed: {exc}")
            raise
        self._workspace_cleaned = True
        self.finished = True

    def close(self) -> None:
        if self._workspace_cleaned:
            return
        if self._pending_status is not None:
            self.finish(self._pending_status)
            return
        self._stop_runner()
        try:
            self.workspace.close()
        except Exception as exc:
            self.record_failure(f"Workspace cleanup failed: {exc}")
            raise
        self._workspace_cleaned = True
        self.finished = True

    def _stop_runner(self) -> None:
        if self._runner_stop_attempted:
            return
        self._runner_stop_attempted = True
        try:
            self.code_runner.stop()
        except Exception as exc:
            self.record_failure(f"Executor cleanup failed: {exc}")


def _bounded_text(text: str, max_bytes: int) -> str:
    payload = text.encode()[:max_bytes]
    return payload.decode(errors="ignore")


def _redact_secrets(text: str) -> str:
    redacted = SECRET_ASSIGNMENT.sub(lambda match: f"{match.group(1)}=[REDACTED]", text)
    return OPENAI_KEY.sub("[REDACTED]", redacted)
