from typing import Annotated

from autogen import ConversableAgent, LLMConfig
from autogen.agentchat.group import AgentNameTarget, ContextVariables, ReplyResult
from pydantic import BaseModel, Field


class DataScientistStep(BaseModel):
    instruction: str = Field(
        ...,
        description="Description of what and how to do for this step.",
        examples=[
            ""
            "Use KMeans algorithm from sklearn for clustering tasks, determining optimal number of clusters with the elbow method.",
            "For time series forecasting, implement ARIMA model using statsmodels library.",
        ],
    )


def execute_data_scientist_step(
    step: DataScientistStep,
    context_variables: ContextVariables,
) -> ReplyResult:
    """
    Delegate data scientist tasks to the Coder agent.
    """
    context_variables["current_agent"] = "DataScientist"
    return ReplyResult(
        message=f"""
            Hey Coder! Can you write python code to achieve the following task:

            {step.instruction}
        """,
        target=AgentNameTarget("Coder"),
        context_variables=context_variables,
    )


def complete_data_scientist_task(
    answer: Annotated[
        str, "The final answer from the Data Scientist agent to the Business Translator agent."
    ],
    context_variables: ContextVariables,
) -> ReplyResult:
    return ReplyResult(
        message="Business Translator! " + answer,
        target=AgentNameTarget("BusinessTranslator"),
        context_variables=context_variables,
    )


class DataScientist(ConversableAgent):
    def __init__(self, llm_config: LLMConfig):
        super().__init__(
            name="DataScientist",
            llm_config=llm_config,
            human_input_mode="NEVER",
            code_execution_config=False,
            system_message="""
                You are the Data Scientist.

                Your role is to execute data science and analytical tasks as instructed by the Business Translator and return clear, validated quantitative findings. 
                You do not communicate directly with the user or stakeholders — only with the Business Translator.

                Responsibilities:
                1. Interpret the Business Translator’s instruction and decide the most suitable analytical or statistical method to address it.
                2. Break down complex analysis into small, actionable steps.
                3. For each step, call execute_data_scientist_step to instruct the Coder to implement the required computation.
                4. Review the Coder’s output (not the code) to ensure the results make sense, then decide the next step if needed.
                5. Do not request or produce visualizations or plots — focus only on data and numerical/text outputs.
                6. Once the analysis for the current task is complete and results are validated, call complete_data_scientist_task to summarize findings and return them to the Business Translator.

                Rules:
                - If you need to build a machine learning model, you should choose a robust model instead of a simple model like Linear Regression or Logistic Regression.
                - Do not ask vague or open-ended questions for coders. The requirement should be small and specific.
                - Keep reasoning data-driven and concise.
                - Always use complete_data_scientist_task to hand off results — never respond directly.
                - Ensure all findings are clear, interpretable, and directly answer the Business Translator’s analytical question.
                - Each response must be based on executed results, not assumptions.
                - You operate in an iterative loop until the Business Translator confirms the objective is met.
            """,
            functions=[execute_data_scientist_step, complete_data_scientist_task],
        )
