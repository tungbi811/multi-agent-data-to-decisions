from autogen import UserProxyAgent
from autogen.agentchat import run_group_chat
from autogen.agentchat.group import AgentTarget, ContextVariables, RevertToUserTarget
from autogen.agentchat.group.patterns import DefaultPattern

from .business_analyst import BusinessAnalyst, DatasetRegistry
from .business_translator import BusinessTranslator
from .coder import Coder
from .config import build_llm_config
from .data_scientist import DataScientist
from .execution import CodeRunner
from .workspace import RunWorkspace


MAX_ROUNDS = 60


class GroupChat:
    def __init__(self, api_key: str, workspace: RunWorkspace, code_runner: CodeRunner):
        context_variables = ContextVariables(
            data={
                "current_agent": "",
                "objective": "",
                "problem_type": "",
                "stakeholders_expectations": [],
                "research_questions": [],
            }
        )
        registry = DatasetRegistry(workspace.root, workspace.dataset_relative_paths)
        business_analyst = BusinessAnalyst(registry, build_llm_config(api_key, 0.5))
        business_translator = BusinessTranslator(build_llm_config(api_key, 0.3))
        data_scientist = DataScientist(build_llm_config(api_key, 0.3))
        coder = Coder(code_runner, build_llm_config(api_key, 0.0))
        user = UserProxyAgent(name="User", code_execution_config=False)

        business_translator.handoffs.set_after_work(RevertToUserTarget())

        self.pattern = DefaultPattern(
            initial_agent=business_analyst,
            agents=[business_analyst, business_translator, data_scientist, coder],
            user_agent=user,
            context_variables=context_variables,
            group_after_work=AgentTarget(business_analyst),
        )

    def run(self, dataset_paths: tuple[str, ...], user_requirements: str):
        message = f"Data paths: {list(dataset_paths)}\nRequirements: {user_requirements}"
        response = run_group_chat(
            pattern=self.pattern,
            messages=message,
            max_rounds=MAX_ROUNDS,
        )
        return iter(response.events)
