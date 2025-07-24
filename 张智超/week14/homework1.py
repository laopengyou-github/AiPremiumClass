import os
from dotenv import load_dotenv, find_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from fastapi import FastAPI
import uvicorn
from langserve import add_routes

if __name__ == '__main__':
    load_dotenv(find_dotenv())

    prompt_temp = ChatPromptTemplate.from_messages(
        [
            ("system", "你是一个熟知{sport}这一运动的AI助理"),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{text}")
        ]
    )
    model = ChatOpenAI(
        model_name="glm-4-flash-250414",
        base_url=os.environ.get('base_url'),
        api_key=os.environ.get('api_key'),
    )
    parser = StrOutputParser()
    chain = prompt_temp | model | parser

    # 保存聊天记录
    store = {}
    def get_session_history(session_id):
        if session_id not in store:
            # 创建一个保存在内存里的聊天记录
            store[session_id] = InMemoryChatMessageHistory()
        return store[session_id]
    with_message_history = RunnableWithMessageHistory(
        chain,
        get_session_history=get_session_history,
        input_messages_key='text',
        history_messages_key='history' # 和MessagesPlaceholder中的variable_name保持一致
    )

    while True:
        user_input = input('请输入问题：')
        if user_input == '离开': break
        content = with_message_history.invoke(
            {'sport': '足球', 'text': user_input},
            config={"configurable": {"session_id": "abc123"}}
        )
        print('AI：', content)


    # app = FastAPI(
    #     title="LangChain API",
    #     description="Sport AI",
    #     version="1.0.0"
    # )

    # # 网络服务注册
    # add_routes(
    #     app,
    #     chain,
    #     path="/sport_ai",
    # )

    # uvicorn.run(app, host='localhost', port=6006)