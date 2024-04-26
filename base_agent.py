from typing import List

from langchain.prompts.chat import (
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain.schema import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)


from langchain_community.chat_models import GigaChat # type: ignore


class BaseAgent:
    def __init__(
        self,
        system_message: SystemMessage,
        model: GigaChat,
        tools: List,
    ) -> None:
        self.system_message = system_message
        self.model = model
        self.tools = tools
        self.init_messages()

    def reset(self) -> None:
        self.init_messages()
        return self.stored_messages # type: ignore

    def init_messages(self) -> None:
        self.stored_messages = [self.system_message]

    def update_messages(self, message: BaseMessage) -> List[BaseMessage]:
        self.stored_messages.append(message) # type: ignore # type: ignore
        return self.stored_messages # type: ignore