from langchain_core.messages import BaseMessage
from pydantic import BaseModel


class ChatbotState(BaseModel):
    messages: list[BaseMessage]
