import os

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage

from src.chatbot import CoreAgent

load_dotenv()

st.set_page_config(page_title="Board Game RAG")
st.title("Board Game RAG")

agent = CoreAgent(api_key=os.environ["API_KEY"])

if "messages" not in st.session_state:
    st.session_state.messages = [
        AIMessage(content="Hello!"),
    ]

for message in st.session_state.messages:
    if isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.write(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.write(message.content)

prompt = st.chat_input("Say Something")

if prompt is not None and prompt != "":
    # add the message to chat message container
    if not isinstance(st.session_state.messages[-1], HumanMessage):
        st.session_state.messages.append(HumanMessage(content=prompt))
        # display to the streamlit application
        message = st.chat_message("user")
        message.write(f"{prompt}")

    if not isinstance(st.session_state.messages[-1], AIMessage):
        with st.chat_message("assistant"):
            # use .write() method for non-streaming, which means .invoke() method in chain
            response = st.write_stream(agent.get_response(prompt, st.session_state.messages))
        st.session_state.messages.append(AIMessage(content=response))
