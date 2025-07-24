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
        temperature=0.3  # 较低的温度使回答更确定性
    )

    # 图书管理员的提示模板
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="""你是一个专业的图书管理员，精通各类书籍信息和图书馆业务。
        你可以:
        1. 查询书籍信息（作者、出版年份、内容简介）
        2. 根据主题或作者推荐书籍
        3. 管理借阅和归还记录
        4. 提供阅读建议和书单

        请简洁、专业地回答用户问题。"""),
        MessagesPlaceholder(variable_name="messages"),
    ])

    # 构建链
    chain = prompt | model

    # 会话存储
    library_sessions = {}

    # 获取会话历史的回调函数
    def get_library_history(session_id):
        if session_id not in library_sessions:
            library_sessions[session_id] = InMemoryChatMessageHistory()
        return library_sessions[session_id]

    # 注入消息历史的链
    library_chain = RunnableWithMessageHistory(
        chain,
        get_session_history=get_library_history,
        input_messages_key="messages"
    )

    # 启动图书馆服务会话
    session_id = input("请输入借阅证ID (例如: lib2023001): ")
    print(f"欢迎回来，借阅证 {session_id} 的持有者。")
    print("你可以询问书籍信息、请求推荐或管理借阅记录。输入 '结束' 退出服务。")

    while True:
        user_input = input("你: ")
        if user_input.lower() == "结束":
            break

        response = library_chain.invoke(
            {"messages": [HumanMessage(content=user_input)]},
            config={"configurable": {"session_id": session_id}}
        )

        print(f"图书管理员: {response.content}")