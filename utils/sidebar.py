import streamlit as st


class Sidebar:
    """
    A class to create and manage the Streamlit sidebar for the application.
    """

    def __init__(self):
        """Initializes the Sidebar class and renders the sidebar."""
        with st.sidebar:
            st.header("⚙️ Settings")
            self._get_api_key()
            # self._get_provider_choice()
            # self._get_model_choice()
            # self._get_temperature()
            self._upload_dataset()
            self._get_user_requirements()
            st.markdown(
                "[View source code](https://github.com/tungbi811/multi-agent-data-to-decisions)"
            )

    def _get_api_key(self):
        """Renders the API key input widget."""
        st.subheader("🔑 API Key")
        self.api_key = st.text_input(
            "OPENAI API Key",
            type="password",
            value="",
            label_visibility="collapsed",
            placeholder="Enter your API key",
        )
        st.markdown("[Get an API key](https://platform.openai.com/account/api-keys)")

    def _get_provider_choice(self):
        """Renders the provider selection widget."""
        st.subheader("🌐 Provider")
        self.provider_choice = st.selectbox(
            "Provider",
            ["OpenAI", "Anthropic", "Azure", "Custom"],
            index=0,
            label_visibility="collapsed",
        )

    def _get_model_choice(self):
        """Renders the model selection widget."""
        st.subheader("🤖 Model")
        self.model_choice = st.selectbox(
            "Model",
            ["gpt-4o-mini", "gpt-4.1-mini", "gpt-3.5-turbo"],
            index=0,  # Default to the newest model
            label_visibility="collapsed",
        )

    def _get_temperature(self):
        """Renders the temperature slider widget."""
        st.subheader("🌡️ Temperature")
        self.temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=0.3,
            step=0.1,
            label_visibility="collapsed",
        )

    def _upload_dataset(self):
        """Renders the file uploader and keeps uploads in UI memory."""
        st.subheader("📂 Dataset")
        self.uploaded_files = st.file_uploader(
            "Upload CSV", type=["csv"], accept_multiple_files=True, label_visibility="collapsed"
        )

    def _get_user_requirements(self):
        """Renders the user requirements input widget."""
        st.subheader("📝 Requirements")
        self.user_requirements = st.text_area(
            "Describe your data analysis requirements here...",
            height=150,
            placeholder="E.g., Analyze sales trends, predict customer churn, etc.",
            # value="Can you segment properties into clusters (luxury homes, affordable starter homes, investment-ready properties, etc.)"
        )
