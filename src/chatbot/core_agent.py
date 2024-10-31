from typing import Any, Iterator

from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, SecretStr


class CoreAgent(BaseModel):
    api_key: SecretStr

    model: str = "gpt-4o-mini"
    temperature: int = 0
    max_tokens: int = 1000

    _llm_client: ChatOpenAI

    def model_post_init(self, __context: Any) -> None:
        self._llm_client = ChatOpenAI(
            api_key=self.api_key,
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

    def get_response(self, user_query: str, conversation_history: list[BaseMessage]) -> Iterator[str]:
        prompt_template = """You are an AI Assistant. Answer the following question considering the history.

        Chat History: {conversation_history}

        User Query: {user_query}"""

        prompt = ChatPromptTemplate.from_template(prompt_template)
        chain = prompt | self._llm_client | StrOutputParser()

        return chain.stream(
            {
                "conversation_history": conversation_history,
                "user_query": user_query,
            }
        )
