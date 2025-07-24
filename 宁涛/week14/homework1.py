import os
from dotenv import load_dotenv, find_dotenv
from langchain_openai import ChatOpenAI
import uvicorn
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import SystemMessage, HumanMessage, trim_messages
from fastapi import FastAPI


if __name__ == '__main__':
    load_dotenv(find_dotenv())
    # template对象
    prompt_temp = ChatPromptTemplate.from_messages(
        [
            ('system', '你是一个图书管理系统，你可以根据图书简介和用户问题进行回答。'),
            MessagesPlaceholder(variable_name='messages')
        ]
    )

    model = ChatOpenAI(
        model="glm-4-flash-250414",
        base_url=os.environ['base_url'],
        api_key=os.environ['api_key'],
        temperature=0.7
    )

    output_parser = StrOutputParser()

    chain = prompt_temp | model | output_parser

    store = {}

    def get_session_hist(session_id):
        if session_id not in store:
            store[session_id] = InMemoryChatMessageHistory()
        return store[session_id]

    with_msg_hist = RunnableWithMessageHistory(
        chain,
        get_session_hist, 
        input_messages_key='messages'
    )

        # session_id
    session_id = "abc123"

    while True:
        # 用户输入
        user_input = input('用户输入的Message：')
        # 调用注入聊天历史的对象
        response = with_msg_hist.invoke(
            input={
                'messages':[HumanMessage(content=user_input)],
                # 'lang':'英语',
                }
            , 
            config={
                'session_id': session_id
            }
        )

        print('AI Message:', response)

