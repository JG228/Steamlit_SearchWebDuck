import streamlit as st
from langchain.agents import initialize_agent, AgentType
from langchain.callbacks import StreamlitCallbackHandler
from langchain.chat_models import ChatOpenAI
from google_search_run import GoogleSearchRun

with st.sidebar:
    openai_api_key = st.secrets["openai"]["api_key"]
    "[Get an OpenAI API key](https://platform.openai.com/account/api-keys)"
    "[View the source code](https://github.com/JG21243/Steamlit_SearchWeb/blob/main/app.py)"

st.title("Josh's Web Powered AI Assistant")

"""
In this example, we're using `StreamlitCallbackHandler` to display the thoughts and actions of an agent in an interactive Streamlit app.
Try more LangChain 🤝 Streamlit Agent examples at [github.com/langchain-ai/streamlit-agent](https://github.com/langchain-ai/streamlit-agent).
"""

system_instruction = "Format your response using emojis, symbols, and formatted text when appropriate."

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "system", "content": system_instruction},
        {"role": "assistant", "content": "Hi, I'm a chatbot who can search the web. How can I help you?"}
    ]

for msg in st.session_state.messages:
    if msg["role"] == "assistant":
        st.markdown(f"**Assistant**: {msg['content']}", unsafe_allow_html=True)
    else:
        st.markdown(f"**User**: {msg['content']}", unsafe_allow_html=True)

if prompt := st.chat_input(placeholder="Who won the Women's U.S. Open in 2018?"):
    st.session_state.messages.append({"role": "user", "content": prompt})

    if not openai_api_key:
        st.info("Please add your OpenAI API key to continue.")
        st.stop()

    llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=openai_api_key, streaming=True)
    search = GoogleSearchRun(name="Search")
    search_agent = initialize_agent([search], llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, handle_parsing_errors=True)
    with st.spinner("Assistant is thinking..."):
        response = search_agent.run(st.session_state.messages, callbacks=[StreamlitCallbackHandler()])
        st.session_state.messages.append({"role": "assistant", "content": response})
        st
