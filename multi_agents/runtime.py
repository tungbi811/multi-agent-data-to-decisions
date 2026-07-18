from __future__ import annotations

import contextlib
import time
from typing import Any

from .execution import CodeRunner, create_docker_executor
from .group_chat import GroupChat
from .workspace import RunWorkspace


MAX_FAILURE_CHARS = 1000


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
        self.workspace.append_trace(
            {
                "type": getattr(event, "type", "unknown"),
                "content": str(content),
            }
        )
        if self._recommendation_recorded:
            return
        if getattr(content, "sender", None) != "BusinessTranslator":
            return
        recommendation = getattr(content, "content", None)
        if not isinstance(recommendation, str):
            return
        bounded = _bounded_text(recommendation, self.workspace.limits.max_record_bytes)
        self.workspace.record_recommendation(bounded)
        self._recommendation_recorded = True

    def finish(self, status: str) -> None:
        if self.finished:
            return
        try:
            self._stop_runner()
            elapsed = time.monotonic() - self.started_at
            try:
                self.workspace.finalize(status, elapsed, self.failures)
            except Exception as exc:
                self._add_failure(f"Workspace finalization failed: {exc}")
                with contextlib.suppress(Exception):
                    self.workspace.close()
        finally:
            self.finished = True

    def close(self) -> None:
        if self.finished:
            return
        try:
            self._stop_runner()
        finally:
            with contextlib.suppress(Exception):
                self.workspace.close()
            self.finished = True

    def _stop_runner(self) -> None:
        try:
            self.code_runner.stop()
        except Exception as exc:
            self._add_failure(f"Executor cleanup failed: {exc}")

    def _add_failure(self, message: str) -> None:
        self.failures.append(message[:MAX_FAILURE_CHARS])


def _bounded_text(text: str, max_bytes: int) -> str:
    payload = text.encode()[:max_bytes]
    return payload.decode(errors="ignore")
