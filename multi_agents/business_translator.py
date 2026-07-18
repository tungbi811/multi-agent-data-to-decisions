from autogen import ConversableAgent, LLMConfig, UpdateSystemMessage
from autogen.agentchat.group import AgentNameTarget, ContextVariables, ReplyResult
from pydantic import BaseModel, Field


class BusinessTranslationStep(BaseModel):
    instruction: str = Field(
        ...,
        description="A specific instruction or task that needs to be solved to answer stakeholder expectations.",
        example=[
            "Calculate the average customer lifetime value for each customer segment.",
            "Identify the top 5 factors contributing to customer churn.",
        ],
    )


def execute_business_translation_step(
    step: BusinessTranslationStep,
    context_variables: ContextVariables,
) -> ReplyResult:
    """
    Translate a high-level business task into specific data science objectives.
    Example task: 'Increase customer retention by 10% over the next quarter.'
    """
    return ReplyResult(
        message=f"""
            Data Scientist, please help execute the following business translation step:
            
            {step.instruction}
        """,
        target=AgentNameTarget("DataScientist"),
        context_variables=context_variables,
    )


class BusinessTranslator(ConversableAgent):
    def __init__(self, llm_config: LLMConfig):
        super().__init__(
            name="BusinessTranslator",
            llm_config=llm_config,
            human_input_mode="NEVER",
            update_agent_state_before_reply=UpdateSystemMessage(
                """
                    Your role is to interpret analytical results from the Data Scientist and translate them into clear, 
                    actionable business recommendations that tell stakeholders exactly what to do and why. 
                    You act as the bridge between technical findings and strategic decision-making.

                    Stakeholder Expectations:
                    {stakeholders_expectations}
                    Research Questions: 
                    {research_questions}

                    Responsibilities:
                    - Interpret the analytical outputs provided by the Data Scientist, focusing on what the numbers mean for business actions.
                    - Translate statistical results into specific recommendations that describe how stakeholders should act, change, or prioritize.
                    - Avoid vague or generic advice (e.g., “adjust strategy,” “use insights”) — always specify the concrete action, threshold, or adjustment implied by the data.
                    - Avoid technical terminology (e.g., “model,” “algorithm,” “regression,” “cluster”) and instead describe outcomes in plain business language.
                    - Tailor recommendations to each stakeholder group’s goals and decision areas.
                    - Ensure that each recommendation includes a rationale backed by the Data Scientist’s findings (what metric or pattern supports it).

                    Workflow:
                    1. Review the {research_questions} and analyze {stakeholders_expectations} to identify key desired outcomes and KPIs.
                    2. For each step in your plan, call execute_business_translation_step to delegate the implementation or computation to the DataScientist agent.
                    3. Continue this iterative process until all research questions have been addressed.
                    4. Once all results are received, interpret and summarize them into actionable, stakeholder-oriented recommendations.
                    5. Present the final recommendations in a structured, statistics-driven, executive-friendly format (e.g., by stakeholder or business theme).

                    Rules:
                    - Always return your final answer in Markdown format with proper headings, bullet points, and emphasis for key points.
                    - Do not include technical details (algorithms, preprocessing, or model design).
                    - Use clear, persuasive, and business-oriented language suitable for executives and decision-makers.
                    - Keep recommendations practical, relevant, and impact-focused.
                    - Always ensure traceability from research question → analytical finding → business recommendation.
                """
            ),
            functions=[execute_business_translation_step],
        )
