from pathlib import Path
from typing import Annotated, List, Literal

import pandas as pd
from autogen import AssistantAgent, LLMConfig
from autogen.agentchat.group import (
    AgentNameTarget,
    ContextVariables,
    ReplyResult,
    RevertToUserTarget,
)
from pydantic import BaseModel, Field


DATA_PREVIEW_ROWS = 5
MAX_PREVIEW_COLUMNS = 10
MAX_DISPLAY_TEXT_CHARS = 80
MAX_COLUMN_LIST_ITEMS = 10
METADATA_CHUNK_ROWS = 1_000
MAX_DATA_INFO_CHARS = 8_000
TRUNCATION_MARKER = "…"


def _bounded_text(value: object) -> str:
    text = str(value).replace("\r", "\\r").replace("\n", "\\n").replace("\t", "\\t")
    if len(text) <= MAX_DISPLAY_TEXT_CHARS:
        return text
    return text[: MAX_DISPLAY_TEXT_CHARS - len(TRUNCATION_MARKER)] + TRUNCATION_MARKER


def _format_column_names(columns: list[object]) -> str:
    displayed = [_bounded_text(column) for column in columns[:MAX_COLUMN_LIST_ITEMS]]
    if len(columns) > MAX_COLUMN_LIST_ITEMS:
        displayed.append(f"{len(columns) - MAX_COLUMN_LIST_ITEMS} more")
    return str(displayed)


class DatasetRegistry:
    def __init__(self, workspace_root: Path, allowed_relative_paths: tuple[str, ...]) -> None:
        self.workspace_root = workspace_root.resolve()
        self.allowed = {
            self._resolve_within_workspace(
                relative,
                error=f"Allowlist entry {relative!r} is not a safe relative path within the workspace",
            )
            for relative in allowed_relative_paths
        }

    def resolve(self, data_path: str) -> Path:
        candidate = self._resolve_within_workspace(
            data_path,
            error=f"{data_path!r} is not an uploaded dataset",
        )
        if candidate not in self.allowed:
            raise ValueError(f"{data_path!r} is not an uploaded dataset")
        return candidate

    def _resolve_within_workspace(self, relative_path: str, *, error: str) -> Path:
        relative = Path(relative_path)
        if relative.is_absolute() or not relative.parts or ".." in relative.parts:
            raise ValueError(error)
        candidate = (self.workspace_root / relative).resolve()
        if not candidate.is_relative_to(self.workspace_root):
            raise ValueError(error)
        return candidate

    def get_data_info(
        self,
        data_path: Annotated[str, "Uploaded dataset path relative to /workspace"],
    ) -> ReplyResult:
        path = self.resolve(data_path)
        with path.open("rb") as dataset_stream:
            header = pd.read_csv(dataset_stream, nrows=0)
            total_columns = len(header.columns)
            inspected_columns = min(total_columns, MAX_PREVIEW_COLUMNS)

            dataset_stream.seek(0)
            preview = pd.read_csv(
                dataset_stream,
                nrows=DATA_PREVIEW_ROWS,
                usecols=range(inspected_columns),
            )

            dataset_stream.seek(0)
            row_count = sum(
                len(chunk)
                for chunk in pd.read_csv(
                    dataset_stream,
                    usecols=[0],
                    chunksize=METADATA_CHUNK_ROWS,
                )
            )

        numerical_columns = preview.select_dtypes(include=["number"]).columns.tolist()
        categorical_columns = preview.select_dtypes(include=["object", "category"]).columns.tolist()
        display_preview = preview.copy()
        display_preview.columns = [_bounded_text(column) for column in display_preview.columns]
        display_preview = display_preview.map(_bounded_text)
        omitted_columns = total_columns - inspected_columns
        omission_note = (
            f"; {omitted_columns} additional columns omitted from preview and type sampling"
            if omitted_columns
            else ""
        )
        dataset_name = _bounded_text(path.stem.replace("_", " ").title())
        message = (
            f"### {dataset_name}\n\n"
            f"Shape: {row_count} rows × {total_columns} columns\n\n"
            f"First five rows (first {inspected_columns} columns{omission_note}):\n\n"
            f"```text\n{display_preview.to_string(index=False)}\n```\n\n"
            "Numerical columns (sample-inferred from the first five rows): "
            f"{_format_column_names(numerical_columns)}\n\n"
            "Categorical columns (sample-inferred from the first five rows): "
            f"{_format_column_names(categorical_columns)}"
        )
        if len(message) > MAX_DATA_INFO_CHARS:
            suffix = "\n\n[Inspection output truncated]"
            message = message[: MAX_DATA_INFO_CHARS - len(suffix)] + suffix
        return ReplyResult(message=message, target=AgentNameTarget("BusinessAnalyst"))


class BizAnalystOutput(BaseModel):
    objective: str = Field(
        ...,
        description=(
            "A clear explanation of the goal of this project for the business. "
            "It should describe the problem being solved, why it matters, and "
            "how it aligns with business strategy."
        ),
        example=(
            "The goal of this project is to implement a predictive model to "
            "forecast customer churn, enabling proactive retention strategies "
            "and reducing revenue loss."
        ),
    )
    stakeholders_expectations: str = Field(
        ...,
        description=(
            "Explain how the results will be used, who will use them, and who will "
            "be impacted by them. Identify both direct users and downstream stakeholders."
        ),
        example=(
            "The marketing team will use the predictions to design retention campaigns. "
            "Customer success managers will use them to prioritize outreach. "
            "Customers may experience more relevant engagement, improving satisfaction."
        ),
    )
    research_questions: List[str] = Field(
        ...,
        description="A list of specific research questions that the analysis aims to answer.",
        example=[
            "What demographic or behavioral characteristics most strongly correlate with churn?",
            "What role do service-related factors (delivery delays, complaints) play in customer attrition?",
            "How does customer engagement correlate with retention rates?",
        ],
    )
    problem_type: Literal[
        "classification",
        "regression",
        "clustering",
        "time_series",
        "anomaly_detection",
        "recommendation",
    ] = Field(
        ...,
        description="The type of machine learning problem that best fits the business use case.",
        example="classification",
    )


def request_clarification(
    clarification_question: Annotated[str, "One targeted question to clarify user requirements"],
) -> ReplyResult:
    """
    Request clarification from the user when the query is ambiguous
    """
    return ReplyResult(
        message=clarification_question,
        target=RevertToUserTarget(),
    )


def complete_business_analyst(
    output: BizAnalystOutput, context_variables: ContextVariables
) -> ReplyResult:
    context_variables["objective"] = output.objective
    context_variables["research_questions"] = output.research_questions
    context_variables["problem_type"] = output.problem_type
    context_variables["stakeholders_expectations"] = output.stakeholders_expectations
    markdown_response = f"""The business analysis is complete with the following details:
            - Objective: {output.objective}
            - Stakeholder Expectations: {output.stakeholders_expectations}
            - Research Questions: {", ".join(output.research_questions)}
            - Problem Type: {output.problem_type}
            """
    return ReplyResult(
        message=markdown_response,
        target=AgentNameTarget("BusinessTranslator"),
        context_variables=context_variables,
    )


class BusinessAnalyst(AssistantAgent):
    def __init__(self, registry: DatasetRegistry, llm_config: LLMConfig):
        super().__init__(
            name="BusinessAnalyst",
            llm_config=llm_config,
            system_message="""
                Your role is to transform user requirements into structured, actionable business analysis outputs. 
                You ensure clarity of the business context, goals, stakeholder expectations, and the research questions 
                that guide exploration.

                Key Responsibilities:
                - Define objectives: List clear, measurable business goals that align with the use case.
                - Formulate research questions: Define the analytical questions that need to be answered to achieve the objectives.
                - Complete business analysis: When you have a clear understanding of the business context, objectives, and research questions, you must call complete_business_analyst to hand off to the BusinessTranslator.
                - Request clarification: If user requirements are ambiguous, incomplete, or conflicting, you must call request_clarification to ask for more details or suggestions and ask for user's choice.

                Workflow:
                1. Review initial user requirements.
                2. call get_data_info first to discover what datasets, variables, and metadata are available for the project.
                3. If requirements are vague, incomplete, or conflicting, call request_clarification to ask for more details or suggestions and ask for user's choice.
                4. When complete, you must call complete_business_analyst to hand off to the BusinessTranslator.

                Rules:
                - Do not propose data cleaning, feature engineering, or modeling directly.
                - Keep analysis high-level, business-focused, and actionable.
                - Don't call get_data_info after the same dataset twice.
            """,
            functions=[registry.get_data_info, request_clarification, complete_business_analyst],
        )
