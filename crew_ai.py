from typing import List, Union


from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage,
)
from creds import GigaChat_creds
from langchain_community.chat_models import GigaChat
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.tools.playwright.extract_text import ExtractTextTool

from langchain.prompts import StringPromptTemplate
from langchain.agents import AgentExecutor, create_react_agent, LLMSingleActionAgent, initialize_agent, Tool

from langchain.schema import AgentAction, AgentFinish
from langchain.chains import LLMChain
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser
import re
from langchain_community.tools.playwright.utils import (
    create_async_playwright_browser,  # A synchronous browser is available, though it isn't compatible with jupyter.\n",      },
)

from base_agent import BaseAgent
from msgs import oracle_messages, writter_messages


class CustomPromptTemplate(StringPromptTemplate):
    # The template to use
    template: str
    # The list of tools available
    tools: List[Tool]
    
    def format(self, **kwargs) -> str:
        # Get the intermediate steps (AgentAction, Observation tuples)
        # Format them in a particular way
        intermediate_steps = kwargs.pop("intermediate_steps")
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\nНаблюдение: {observation}\nМысль: "
        # Set the agent_scratchpad variable to that value
        kwargs["agent_scratchpad"] = thoughts
        # Create a tools variable from the list of tools provided
        kwargs["tools"] = "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])
        # Create a list of tool names for the tools provided
        kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])
        return self.template.format(**kwargs)

class CustomOutputParser(AgentOutputParser):
    
    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        # Check if agent should finish
        if "Final Answer:" in llm_output:
            return AgentFinish(
                # Return values is generally always a dictionary with a single `output` key
                # It is not recommended to try anything else at the moment :)
                return_values={"output": llm_output.split("Final Answer:")[-1].strip()},
                log=llm_output,
            )
        # Parse out the action and action input
        regex = r"Мысль:(.*?)\s*Действие:(.*?)\s*Наблюдение:[\s]*(.*)"
        match = re.search(regex, llm_output, re.DOTALL)
        if not match:
            raise ValueError(f"Could not parse LLM output: `{llm_output}`")
        action = match.group(1).strip()
        action_input = match.group(2)
        # Return the action and action input
        return AgentAction(tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output)

class TWriterAgent(BaseAgent):
    def __init__(
        self,
        system_message: SystemMessage,
        model: GigaChat,
        tools: List,
    ) -> None:
        super().__init__(system_message=system_message, model=model, tools=tools)
        pass

    def step(
        self,
        input_message: HumanMessage,
    ) -> AIMessage:
        messages = self.update_messages(input_message)

        output_message = self.model.invoke(messages)
        res = self.tools[0](output_message.content)
        self.update_messages(output_message)
        return output_message, res



class AIAssistant:
    def __init__(self):

        self.giga = GigaChat(
            model="GigaChat-Pro",
            credentials=GigaChat_creds,
            profanity_check=False,
            temperature=1,
            verify_ssl_certs=False,
            scope="GIGACHAT_API_CORP"
        )

        self.search_tool = DuckDuckGoSearchResults()
        self.async_browser = create_async_playwright_browser()
        self.extractor = ExtractTextTool(async_browser=self.async_browser)


        self.t_writer_tools= [self.search_tool]

        self.oracle_tools = [
            Tool(
                name="Поисковик", 
                func = self.search_tool, 
                description = "Инструмент-строка для поиска по полученному запросу"), 
            Tool(
                name="Экстрактор", 
                func = self.extractor, 
                description = "Инструмент для вытягивания текста по ссылке"),
            ]


        self.oracle_messages = oracle_messages
        self.writter_messages = writter_messages

        self.t_writer = TWriterAgent(model=self.giga, tools=self.t_writer_tools, system_message=self.writter_messages[0])

        oracle_prompt = CustomPromptTemplate(
            template=self.oracle_messages,
            tools=self.oracle_tools,
            # This omits the `agent_scratchpad`, `tools`, and `tool_names` variables because those are generated dynamically
            # This includes the `intermediate_steps` variable because that is needed
            input_variables=["input", "intermediate_steps"]
        )
        self.output_parser = CustomOutputParser()
        self.oracle_chain = LLMChain(llm=self.giga, prompt=oracle_prompt)

        self.oracle_tool_names = [tool.name for tool in self.oracle_tools]

        self.agent = LLMSingleActionAgent(
            llm_chain=self.oracle_chain, 
            output_parser=self.output_parser,
            stop=["\nВывод: "], 
            allowed_tools=[tool.name for tool in self.oracle_tools]
        )
        self.oracle_executor = AgentExecutor.from_agent_and_tools(agent=self.agent, tools=self.oracle_tools, verbose=True)

    def __call__(self, user_task):
        user_msg = HumanMessage(content=user_task)

        mes, link = self.t_writer.step(user_msg)

        print(f"User: {user_msg.content}")
        print(f"Writter: {mes.content}")

        self.oracle_executor.run(mes)


if __name__ == "__main__":
    assistant = AIAssistant()
    msg = str(input())
    assistant(msg)