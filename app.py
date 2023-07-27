import streamlit as st
from langchain.agents import initialize_agent, AgentType
from langchain.callbacks import StreamlitCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.tools import DuckDuckGoSearchRun

# Set custom theme
st.set_page_config(
    page_title="Josh's Web Powered AI Assistant",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon=":robot_face:",
    theme={
        "primaryColor": "#4f8bf9",
        "backgroundColor": "#f0f2f6",
        "secondaryBackgroundColor": "#e0e0e0",
        "textColor": "#000000",
        "font": "sans-serif",
    },
)

with st.sidebar:
    st.title("Settings")
    openai_api_key = st.secrets["openai"]["api_key"]
    st.markdown("[Get an OpenAI API key](https://platform.openai.com/account/api-keys)", unsafe_allow_html=True)
    st.markdown("[View the source code](https://github.com/JG21243/Steamlit_SearchWeb/blob/main/app.py)", unsafe_allow_html=True)

st.title("Josh's Web Powered AI Assistant")

"""
In this example, we're using `StreamlitCallbackHandler` to display the thoughts and actions of an agent in an interactive Streamlit app.
Try more LangChain 🤝 Streamlit Agent examples at [github.com/langchain-ai/streamlit-agent](https://github.com/langchain-ai/streamlit-agent).
"""

system_instruction = "Format your response using emojis, symbols, and formatted text when appropriate."

# Initialize the chat messages
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "system", "content": system_instruction},
        {"role": "assistant", "content": "Hi, I'm a chatbot who can search the web. How can I help you?"}
    ]

# Display the chat messages
for msg in st.session_state.messages:
    if msg["role"] == "assistant":
        st.markdown(f"**Assistant**: {msg['content']}", unsafe_allow_html=True)
    else:
        st.markdown(f"**User**: {msg['content']}", unsafe_allow_html=True)

# Get user input
if prompt := st.chat_input(placeholder="Type your message here...", help="Enter your question or message for the AI assistant."):
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Check for OpenAI API key
    if not openai_api_key:
        st.info("Please add your OpenAI API key to continue.")
        st.stop()

    # Initialize the LangChain agent and run the chat
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=openai_api_key, streaming=True)
    search = DuckDuckGoSearchRun(name="Search")
    search_agent = initialize_agent([search], llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, handle_parsing_errors=True)

    @st.cache(show_spinner="Assistant is thinking...")
    def run_agent(messages):
        return search_agent.run(messages, callbacks=[StreamlitCallbackHandler()])

    with st.spinner("Assistant is thinking..."):
        response = run_agent(st.session_state.messages)
        st.session_state.messages.append({"role": "assistant", "content": response})
