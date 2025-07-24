import os
from dotenv import load_dotenv, find_dotenv
from langchain_openai import ChatOpenAI #LLM调用的封装
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from fastapi import FastAPI
from langserve import add_routes
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage #对话角色，user, assistant, system
import uvicorn

# 1.导入必要包
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
#  导入template中传递聊天历史信息的”占位“类
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

if __name__ == '__main__':
    

    load_dotenv(find_dotenv())

    model = ChatOpenAI(
        model="GLM-4-Flash-250414",
        base_url=os.environ['base_url'],
        api_key=os.environ['api_key'],
        # temperature=0.7,
    )
    # 带有占位符的prompt template
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system","你是资深开发工程师。尽你所能使⽤{lang}回答所有问题。"),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )

    # LLM结果解析器对象
    parser = StrOutputParser()

    # langchain的表达式语言（LCEL）构建chain
    chain =  prompt | model | parser


    # 定制存储消息的dict
    # key/value: 
    # key: sessionid会话id(资源编号) ,区分不同的用户或者不同的聊天内容
    # value: InMemoryChatMessageHistory存储聊天信息list对象
    store = {}

    # 定义函数，根据sessionId获取聊天历史
    # callback 系统调用时被执行的代码
    def get_session_hist(session_id):
        # 以sessionid为key从store中提取关联的历史聊天对象
        if session_id not in store:
            store[session_id] = InMemoryChatMessageHistory()
        return store[session_id] #检索（langchain负责维护聊天历史信息）

    # 在chain中注入聊天历史信息
    # 调用chain之前，还需要根据sessionid提取不同的聊天历史
    with_msg_hist = RunnableWithMessageHistory(chain, get_session_history=get_session_hist,
                                               input_messages_key="messages") #input_messages_key：指明用户消息使用的key

    session_id = "123"

    while True:
        # 用户输入
        user_input= input('用户输入的message：')
        # 调用注入聊天历史的对象
        response = with_msg_hist.invoke(
            {
                "messages":[HumanMessage(content=user_input)],
                "lang":"英文"
            },
            config={'configurable':{'session_id':session_id}}
        )
        print('AI message:', response)
