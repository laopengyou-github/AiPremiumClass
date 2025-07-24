from dotenv import find_dotenv, load_dotenv
import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory

if __name__ == '__main__':
    load_dotenv(find_dotenv())

    # 初始化LLM模型
    model = ChatOpenAI(
        model="gpt-3.5-turbo",
        api_key=os.environ['OPENAI_API_KEY'],
        temperature=0.7
    )

    # 聊天机器人的提示模板
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="你是一个友好、智能的聊天机器人，可以进行各种话题的交流。"),
        MessagesPlaceholder(variable_name="messages"),
    ])

    # 构建链
    chain = prompt | model

    # 会话存储
    session_store = {}

    # 获取会话历史的回调函数
    def get_session_history(session_id):
        if session_id not in session_store:
            session_store[session_id] = InMemoryChatMessageHistory()
        return session_store[session_id]

    # 注入消息历史的链
    chat_chain = RunnableWithMessageHistory(
        chain,
        get_session_history=get_session_history,
        input_messages_key="messages"
    )

    # 启动对话
    session_id = input("请输入会话ID (例如: user123): ")
    print(f"会话 {session_id} 已启动。输入 '退出' 结束对话。")

    while True:
        user_input = input("你: ")
        if user_input.lower() == "退出":
            break

        response = chat_chain.invoke(
            {"messages": [HumanMessage(content=user_input)]},
            config={"configurable": {"session_id": session_id}}
        )

        print(f"机器人: {response.content}")