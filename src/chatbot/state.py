from langchain_core.messages import BaseMessage
from pydantic import BaseModel


class BoardGame(BaseModel):
    name: str
    publish_year: int
    designer: str
    num_players_min: int
    num_players_max: int
    description: str


class ChatbotState(BaseModel):
    messages: list[BaseMessage]
    user_query: str
    board_game: BoardGame | None = None
