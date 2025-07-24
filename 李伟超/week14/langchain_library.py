from dotenv import find_dotenv, load_dotenv
import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, HumanMessagePromptTemplate

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableMap
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage


def new_func(user_input):
    return HumanMessage(user_input)


if __name__ == "__main__":
    load_dotenv(find_dotenv())

    # 1、构建模型
    model = ChatOpenAI(
        model="glm-4-flash",
        base_url=os.environ['base_url'],
        api_key=os.environ['api_key'],
        temperature=0.7
    )
    # 2、构建prompt
    prompt = ChatPromptTemplate.from_messages(
        [
        ("system", "你是一个图书管理系统，通过图书简介为借阅读者提供咨询"),
        MessagesPlaceholder(variable_name="messages"),
        ]
    )

    parser = StrOutputParser()

    chain = prompt | model | parser

    store = {}

    def get_session_hist(session_id):
        if session_id not in store:
            store[session_id] = InMemoryChatMessageHistory()
        return store[session_id]
    
    with_msg_hist = RunnableWithMessageHistory(
        chain,
        get_session_history = get_session_hist,
        input_messages_key="messages",
        
    )

    session_id = "abc123"

    while True:
        user_input = input('用户输入的Message：')
        response = with_msg_hist.invoke(
            {
                "messages": [HumanMessage(content=user_input)],
                # "lang": '计算机'
            },
            config = { "configurable": {'session_id': session_id}}
            )
        
        print(response)
   