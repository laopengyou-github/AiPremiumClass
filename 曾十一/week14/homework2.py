from dotenv import find_dotenv, load_dotenv
import os
import sqlite3
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import FileChatMessageHistory

# ✅ 设置统一的历史保存目录
HISTORY_DIR = "/mnt/data_1/zfy/homework/history"
os.makedirs(HISTORY_DIR, exist_ok=True)

# ✅ 历史记录函数（按 session_id 保存）
def get_session_hist(session_id: str):
    history_file = os.path.join(HISTORY_DIR, f"history_{session_id}.json")
    return FileChatMessageHistory(history_file)

# ✅ 从 SQLite 读取图书信息
def load_books_from_db(db_path="/mnt/data_1/zfy/homework/books.db"):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT id, title, author, category, summary FROM books WHERE available = 1")
    rows = cursor.fetchall()
    conn.close()
    books = []
    for row in rows:
        books.append({
            "编号": row[0],
            "书名": row[1],
            "作者": row[2],
            "类型": row[3],
            "简介": row[4]
        })
    return books

# ✅ 构建图书系统 prompt（作为 system message 注入）
def build_book_prompt(books):
    books_info = "\n".join([
        f"{b['编号']}《{b['书名']}》, 作者：{b['作者']}, 类型：{b['类型']}, 简介：{b['简介']}"
        for b in books
    ])
    system_prompt = (
        "你是一个图书馆咨询助手，以下是可借阅图书信息：\n"
        f"{books_info}\n"
        "请根据用户提问，结合图书简介和对话上下文，给出专业简洁的推荐或解答。"
    )
    return system_prompt

if __name__ == '__main__':
    load_dotenv(find_dotenv())

    # ✅ 初始化模型
    model = ChatOpenAI(
        model="glm-4-flash-250414",
        base_url=os.environ['base_url'],
        api_key=os.environ['api_key'],
        temperature=0.7
    )

    # ✅ 构建解析器 + 基础链（prompt + model + parser）
    parser = StrOutputParser()

    # ✅ 加载图书信息并构建 system prompt
    books = load_books_from_db()
    system_prompt_text = build_book_prompt(books)

    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=system_prompt_text),
        MessagesPlaceholder(variable_name="messages"),
    ])

    chain = prompt | model | parser

    # ✅ 封装带历史对话能力的 Chain
    with_msg_hist = RunnableWithMessageHistory(
        chain,
        get_session_history=get_session_hist,
        input_messages_key="messages"
    )

    # ✅ 启动交互对话
    session_id = input("请输入会话 session_id（用于区别多个用户）：").strip()
    print("欢迎来到智能图书馆，输入 quit 结束对话。")

    while True:
        user_input = input("你想咨询什么图书？")
        if user_input.strip().lower() == "quit":
            break

        response = with_msg_hist.invoke(
            {
                "messages": [HumanMessage(content=user_input)],
                "lang": "中文"
            },
            config={"configurable": {"session_id": session_id}}
        )

        print("图书助手：", response)
