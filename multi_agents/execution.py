from __future__ import annotations

import os
from typing import Annotated, Protocol

from autogen.agentchat.group import (
    AgentNameTarget,
    ContextVariables,
    ReplyResult,
    RevertToUserTarget,
)
from autogen.coding import CodeBlock, DockerCommandLineCodeExecutor

from .workspace import RunWorkspace


EXECUTOR_IMAGE = "auto-ds-executor:0.1"
MAX_OUTPUT_CHARS = 12_000


class Executor(Protocol):
    def execute_code_blocks(self, blocks: list[CodeBlock]): ...

    def restart(self) -> None: ...

    def stop(self) -> None: ...


def create_docker_executor(workspace: RunWorkspace) -> DockerCommandLineCodeExecutor:
    return DockerCommandLineCodeExecutor(
        image=EXECUTOR_IMAGE,
        timeout=300,
        work_dir=workspace.root,
        execution_policies={"python": True, "bash": False, "shell": False, "sh": False},
        container_create_kwargs={
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
        },
    )


class CodeRunner:
    def __init__(self, workspace: RunWorkspace, executor: Executor) -> None:
        self.workspace = workspace
        self.executor = executor

    def run_code(
        self,
        code: Annotated[str, "Complete Python script to run in Docker"],
        context_variables: ContextVariables,
    ) -> ReplyResult:
        code_path = self.workspace.record_code(code)
        block = CodeBlock(language="python", code=code)
        try:
            result = self.executor.execute_code_blocks([block])
        except Exception:
            self.executor.restart()
            try:
                result = self.executor.execute_code_blocks([block])
            except Exception as exc:
                message = f"Execution failed after retry: {exc}"
                self.workspace.record_output(code_path.stem, message)
                return ReplyResult(message=message, target=RevertToUserTarget())

        output = str(result.output)
        self.workspace.record_output(code_path.stem, output)
        bounded = output[-MAX_OUTPUT_CHARS:]
        if result.exit_code == 0:
            target_name = context_variables["current_agent"] or "DataScientist"
            return ReplyResult(message=f"Output:\n{bounded}", target=AgentNameTarget(target_name))
        return ReplyResult(message=bounded, target=AgentNameTarget("Coder"))

    def stop(self) -> None:
        self.executor.stop()
