from typing import Any

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, PrivateAttr, SecretStr

from .state import ChatbotState


class Chatbot(BaseModel):
    api_key: SecretStr

    model: str = "gpt-4o-mini"
    temperature: int = 0
    max_tokens: int = 1000

    _llm_client: ChatOpenAI = PrivateAttr()
    _graph: Any = PrivateAttr()
    _memory: MemorySaver = MemorySaver()

    def model_post_init(self, __context: Any) -> None:
        self._llm_client = ChatOpenAI(
            api_key=self.api_key,
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        # Define graph
        graph_compiler = StateGraph(ChatbotState)
        graph_compiler.add_node("chatbot", self.invoke_chatbot)

        graph_compiler.add_edge(START, "chatbot")
        graph_compiler.add_edge("chatbot", END)

        self._graph = graph_compiler.compile(checkpointer=self._memory)

    def invoke_chatbot(self, state: ChatbotState) -> ChatbotState:
        user_query = state.messages[-1].content
        prompt = f"""You are an expert in board games. Extract the name of the board game relevant to this user query.

        User Query: {user_query}"""

        result = self._llm_client.invoke(prompt)
        state.messages.append(result)

        return state

    def __call__(self, user_query: str) -> str:
        config = {
            "configurable": {
                "thread_id": "1",
            },
        }

        init_state = ChatbotState(messages=[HumanMessage(content=user_query)], user_query=user_query)
        response = self._graph.invoke(init_state, config)
        return response["messages"][-1].content
