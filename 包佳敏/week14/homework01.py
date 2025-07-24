from dotenv import find_dotenv, load_dotenv
from langchain_openai import ChatOpenAI #大模型调用封装LLM
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
# 导入template中传递聊天历史信息的“占位”类
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from fastapi import FastAPI
from langserve import add_routes
import uvicorn
import os

if __name__ == "__main__":
    load_dotenv(find_dotenv())

    model = ChatOpenAI(
        model="glm-4-flash-250414",
        openai_api_key=os.environ["api_key"],
        openai_api_base=os.environ["base_url"]
    )

    # 带有占位符的prompt template
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "你是一个精通动漫的AI助手。尽你所能使用{lang}回答所有问题。"),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
    parser = StrOutputParser()
    chain = prompt | model | parser

    #定制存储消息的dict
    store = {}

    # 定义函数：根据sessionId获取聊天记录（callback回调）
    def get_seesion_hist(session_id):
        if session_id not in store:
            store[session_id] = InMemoryChatMessageHistory() 
        return store[session_id] 

    #在chain中注入聊天历史消息
    with_msg_hist = RunnableWithMessageHistory(
        chain, 
        get_session_history=get_seesion_hist,
        input_messages_key="messages")#input_messages_key指明用户消息使用key
    session_id = "jiamin"
    
    while True:
        user_input = input("和我聊聊动漫吧：")
        if user_input.lower() == 'exit':
            break
        response = with_msg_hist.invoke(
            {
                "messages":[HumanMessage(content=user_input)],
                "lang": "粤语",  # 语言参数
            },
            config={'configurable':{'session_id':session_id}})
        print('AI Message:', response)
