from autogen import AssistantAgent, LLMConfig

from .execution import CodeRunner


class Coder(AssistantAgent):
    def __init__(self, code_runner: CodeRunner, llm_config: LLMConfig):
        super().__init__(
            name="Coder",
            llm_config=llm_config,
            human_input_mode="NEVER",
            system_message="""
                You are the Coder agent. Produce complete, reproducible Python scripts.
                Every script runs in a fresh process inside Docker with /workspace as its
                working directory. Read uploaded CSVs using the relative paths supplied in
                the conversation. Do not access the network, environment secrets, or paths
                outside /workspace. Do not install packages. Do not create plots or images.
                Always print the results needed by the Data Scientist. Always call run_code.
                When a result reports an error, correct the complete script and try again.
            """,
            functions=[code_runner.run_code],
        )
