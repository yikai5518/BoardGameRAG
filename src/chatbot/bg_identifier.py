from langchain_openai import ChatOpenAI
from pydantic import BaseModel, PrivateAttr

from .state import BoardGame, ChatbotState


class BGResponse(BaseModel):
    found: bool
    board_game: BoardGame | None


class BoardGameIdentifier(BaseModel):
    llm_client: ChatOpenAI
    _role_description: str = PrivateAttr()

    def model_post_init(self, __context):
        self._role_description = "You are an expert in board games, and you are familiar with the author, rules, and descriptions of all board games."  # noqa: E501

    def __call__(self, input_state: ChatbotState) -> ChatbotState:
        user_query = f"""Given the user query: {input_state.user_query}

        Action: Find all board games mentioned in this query.
            - Pay attention to board game names that might be slightly mispelled. Correct and extract them.
            - According to the name, find the following information: publication year, designer's name, number of players, and description.
        Outcome: A list of the board game names mentioned in this query, and their corresponding descriptions.
            - Only extract the name of the board game e.g. Ark Nova, Brass: Birmingham.
            - If the name is incomplete, find the full name of the game e.g. Dune -> Dune: Imperium.
            - If you cannot correctly identify what board game it is, set found = False, and let board_game be None.
            - If you can identify the board game, output found = True and the board game's name and descritptions: e.g.
                - name: Ark Nova
                - publish_year: 2021
                - designer: Mathias Wigge
                - num_players_min: 1
                - num_players_max: 4
                - description: Plan and build a modern, scientifically managed zoo to support conservation projects.
        """  # noqa: E501

        messages = [{"role": "system", "content": self._role_description}, {"role": "user", "content": user_query}]
        response: BGResponse = self.llm_client.with_structured_output(BGResponse).invoke(messages)  # type: ignore

        if response.found:
            input_state.board_game = response.board_game

        return input_state
