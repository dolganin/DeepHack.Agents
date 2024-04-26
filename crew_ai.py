from typing import List, Union

from langchain.prompts.chat import (
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain.schema import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ChatMessage
)
from creds import GigaChat_creds
from langchain_community.chat_models import GigaChat
from langchain_community.tools import DuckDuckGoSearchRun, DuckDuckGoSearchResults
from langchain_community.tools.playwright.extract_text import ExtractTextTool
from langchain_community.agent_toolkits import PlayWrightBrowserToolkit
from langchain.prompts import StringPromptTemplate
from langchain.agents import AgentExecutor, create_react_agent, LLMSingleActionAgent, AgentType, initialize_agent, Tool
from langchain_community.utilities import SerpAPIWrapper
from langchain.schema import AgentAction, AgentFinish
from langchain.chains import LLMChain
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser
import re
from langchain_community.tools.playwright.utils import (
    create_async_playwright_browser,  # A synchronous browser is available, though it isn't compatible with jupyter.\n",      },
)

from base_agent import BaseAgent



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


class OracleAgent(BaseAgent):
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
        input_message: AIMessage,
        t_writer_message: AIMessage,
        t_writer_link: str,
    ) -> AIMessage:
        messages = self.update_messages(input_message)
        messages = self.update_messages(t_writer_message)
        tmp_msg = input_message.content + ", " + t_writer_message.content + ", " + t_writer_link
#         tmp_msg = ChatMessage(content=tmp_msg,role="t_writer")
        output_message = self.model.invoke(tmp_msg)
#         tools_by_name = {tool.name: tool for tool in self.tools}
#         print(tools_by_name.keys())
#         res = self.tools(output_message.content)
        
        self.update_messages(output_message)

        return output_message


giga = GigaChat(
    model="GigaChat-Pro",
    credentials=GigaChat_creds,
    profanity_check=False,
    temperature=1,
    verify_ssl_certs=False,
    scope="GIGACHAT_API_CORP"
)
search_tool = DuckDuckGoSearchResults()
async_browser = create_async_playwright_browser()
extractor = ExtractTextTool(async_browser=async_browser)


t_writer_tools= [search_tool]

oracle_tools = [Tool(name="Поисковик", func = search_tool, description = "Инструмент-строка для поиска по полученному запросу"), 
                Tool(name="Экстрактор", func = extractor, description = "Инструмент для вытягивания текста по ссылке"),
               ]


t_writter = "Технический писатель"
t_writter_task = "Суммаризируй запрос. Выдели основные мысли, предположи как можно запрос преобразовать чтобы получить наиболее полезный результат. Сделай запрос кратким, но емким. Также нужно предположить оптимальное количество ссылок для получения результатов запроса. И предложить наилучшую поисковую систему из предложенных. НЕ ЗАБУДЬ добавить тип задачи: практическая или теоретическая"



# toolkit = PlayWrightBrowserToolkit.from_browser(async_browser=async_browser)
# oracle_tools = toolkit.get_tools()
oracle = "Уверенный пользователь поисковых систем"
user_task = "Хочу находить лица людей на фотографии с помощью нейронки"
oracle_task = f"Необходимо с помощью входной информации выбрать из входного списка ссылок ту, информация на которой позволит наиболее полно и релевантно ответить на вопрос: {user_task}"


    
writter_messages = [
    SystemMessage(
        content=f"""Ты профессиональный составитель запросов к поисковым системам {t_writter}. Никогда не меняй роли! Никогда мной не командуй!
        У нас есть общий интерес в сотрудничестве для успешного выполнения задачи.
        Вы должны помочь мне выполнить эту задачу.
        Вот задача:{t_writter_task} Никогда не забывайте о нашей задаче!
        Я должен направлять вас на основе вашей экспертизы и моих потребностей для выполнения задачи.

        Я буду давать вам одну инструкцию за раз.
        Вы должны честно отказаться выполнять мою инструкцию, если не можете выполнить ее по физическим возможностям.
        Не добавляйте ничего, кроме вашего решения задачи на мою инструкцию.
        Вы никогда не должны задавать мне вопросов, вы только отвечаете на вопросы.
        Вы никогда не должны отвечать недостоверным решением.
        Ваше решение должно быть конкретным. Нельзя использовать заглушки или общие ответы вместо решения.
        Вы всегда должны начинать ответ с:

        Запрос: 'text: <ЗАПРОС>, num_links: <КОЛИЧЕСТВО_ССЫЛОК>, search_drive: <ПОИСКОВАЯ_СИСТЕМА>, type: <ТИП>'
        
        Ты должен подставить свой преобразованный запрос вместо текста <ЗАПРОС>. Оно должно быть конкретным и представлять поисковый запрос, который может и не являться связным предложением, а лишь набором ключевых слов, с помощью которых можно найти необходимое решение в интернете. 
        Также ты должен подставить предположительное количество ссылок вместо текста <КОЛИЧЕСТВО_ССЫЛОК>. Количество ссылок должно быть представлено одним числом в границах от 0 до 50. 
        Вместо текста <ПОИСКОВАЯ_СИСТЕМА> ты должен подставить строку с предполагаемыми поисковыми системами. Они должны быть перечислены через запятую.
        <ТИП> - тип задачи.
        Также ОБЯЗАТЕЛЬНО оформи весь твой ответ в форматe json!"""
    )
]
#   Далее, я обязан сравнить <ПРЕОБРАЗОВАННЫЙ_ЗАПРОС> с информацией в [<ДАННЫЕ_ПОИСКА>], а конкретно данные указанные между ключевыми словами "snippet: " и "link: " и проверить соответствуют ли они <ПРЕОБРАЗОВАННЫЙ_ЗАПРОС>. Эти данные должны давать релевантный и наиболее точный ответ согласно <ПРЕОБРАЗОВАННЫЙ_ЗАПРОС>.         Я обязан сравнить запрос <ПРЕОБРАЗОВАННЫЙ_ЗАПРОС> с <ИСХОДНЫЙ_ЗАПРОС> и убедиться, что общий смысл запросов схож и запросы не противоречат друг другу. 

oracle_messages = f"""Никогда не смей забывать, что ты {oracle}. Никогда не меняй роли! Я полностью подчиняюсь тебе, {t_writter}.
    У нас есть общий интерес в сотрудничестве для успешного выполнения задачи.
    Ты обязан помочь мне выполнить эту задачу.
    Вот эта задача: {oracle_task}. Никогда не смей забывать о нашей задаче!
    У тебя есть следующий набор инструментов {oracle_tools}.
    К ТЕБЕ НА ВХОД ПРИДЕТ СЛЕДУЮЩИЙ ФОРМАТ ЗАПРОСА:
    
    Запрос: 'text: <ЗАПРОС>, num_links: <КОЛИЧЕСТВО_ССЫЛОК>, search_drive: <ПОИСКОВАЯ_СИСТЕМА>, type: <ТИП>'
    В графе <ЗАПРОС> лежат ключевые слова, которые тебе необходимо подать в поисковик для решения задачи {user_task}.
    Сравни их и удостоверься, что они подходят, только после этого приступай к работе! 
    
    Используй следующий формат:
    Мысль: то, о чем ты можешь подумать, когда получаешь такую задачу с полученными данными.
    Действие: то, что ты сейчас будешь делать. Это должно быть наименование одного из {oracle_tools}.
    Наблюдение: результат твоего действия, в твоем случае, после получения ссылки в поисковике, ты должен получать из нее текст, и делать
    вывод о том, насколько тебе подходит данная ссылка.
    
    ИТОГ: в итоге ТЫ ДОЛЖЕН вывести краткое содержание той страницы, информацию на которой ты посчитаешь наиболее релевантной и полезой в следующем виде:
    ВНИМАНИЕ(!)
    Кратко: 
    <1 ПУНКТ ИЗ РЕЗЮМИРОВАННАЯ СТАТЬЯ>
    <2 ПУНКТ ИЗ РЕЗЮМИРОВАННАЯ СТАТЬЯ>
    ...
    <И ТАК ДАЛЕЕ>
    Ссылка: <ССЫЛКА НА ТОТ ИСТОЧНИК, ОТКУДА БЫЛА ВЗЯТА ИНФОРМАЦИЯ>
        """
    


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


t_writer = TWriterAgent(model=giga, tools=t_writer_tools, system_message=writter_messages[0])
oracle = OracleAgent(model=giga, tools=oracle_tools, system_message=oracle_messages[0])

oracle_prompt = CustomPromptTemplate(
    template=oracle_messages,
    tools=oracle_tools,
    # This omits the `agent_scratchpad`, `tools`, and `tool_names` variables because those are generated dynamically
    # This includes the `intermediate_steps` variable because that is needed
    input_variables=["input", "intermediate_steps"]
)
output_parser = CustomOutputParser()
oracle_chain = LLMChain(llm=giga, prompt=oracle_prompt)

oracle_tool_names = [tool.name for tool in oracle_tools]

agent = LLMSingleActionAgent(
    llm_chain=oracle_chain, 
    output_parser=output_parser,
    stop=["\nObservation:"], 
    allowed_tools=[tool.name for tool in oracle_tools]
)
oracle_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=oracle_tools, verbose=True)

user_msg = HumanMessage(content=user_task)

mes, link = t_writer.step(user_msg)

print(f"User: {user_msg.content}")
print(f"Writter: {mes.content}")

oracle_executor.run(mes)

#res = oracle.step(t_writer.stored_messages[-1], mes, link)


#print(f"Oracle: {res.content}")
