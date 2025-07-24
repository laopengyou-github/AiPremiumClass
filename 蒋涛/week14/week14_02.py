"""
借助langchain实现图书管理系统开发扩展，通过图书简介为借阅读者提供咨询。

- 读取外部csv文件来加载图书信息
- 使用langchain的LLM模型进行图书简介的生成和查询
- 支持根据聊天历史来调整图书咨询的内容
"""

import os
import pandas as pd
import json
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage, AIMessage

# 加载.env文件，获取环境变量
load_dotenv()
# 从环境变量中获取API密钥
API_KEY = os.getenv("API_KEY")
# 从环境变量中获取API基础URL
BASE_URL = os.getenv("BASE_URL")

# 初始化LLM（大语言模型）
llm = ChatOpenAI(
    openai_api_key=API_KEY,  # 设置OpenAI API密钥
    openai_api_base=BASE_URL,  # 设置API基础URL
    model_name="glm-4-flash-250414",  # 可根据实际情况更换模型名称
    temperature=0.7  # 设置模型生成文本的随机性
)

# 读取图书信息，从指定的CSV文件中加载图书数据
def load_books(csv_path):
    # 使用pandas读取CSV文件
    df = pd.read_csv(csv_path)
    # 将DataFrame转换为字典列表，每个字典代表一本图书
    books = df.to_dict(orient="records")
    return books

# 生成图书简介，如果图书简介为空，则调用LLM生成简介
def generate_book_intro(book):
    # 检查图书简介是否为空或缺失
    if pd.isna(book["简介"]) or not book["简介"]:
        # 构造生成简介的提示词
        prompt = f"请为以下图书生成简短简介：书名《{book['书名']}》，作者：{book['作者']}。"
        # 调用LLM生成简介
        response = llm.invoke([HumanMessage(content=prompt)])
        return response.content
    # 如果简介不为空，直接返回原简介
    return book["简介"]

# 图书咨询对话，根据用户查询和聊天历史提供图书咨询
def consult_book(books, user_query, chat_history=None):
    # 如果聊天历史为空，初始化为空列表
    if chat_history is None:
        chat_history = []
    # 拼接所有图书信息，去掉库存字段
    books_info = "\n".join([
        f"{b['编号']}《{b['书名']}》, 作者：{b['作者']}, 类型：{b['类型']}, 简介：{generate_book_intro(b)}"
        for b in books
    ])
    # 构造系统提示词
    system_prompt = (
        "你是一个图书馆咨询助手，以下是图书信息：\n"
        f"{books_info}\n"
        "请根据用户提问，结合图书简介和聊天历史，给出专业、简明的推荐或解答。"
    )
    # 初始化消息列表，添加系统提示
    messages = [SystemMessage(content=system_prompt)]
    # 遍历聊天历史，将消息添加到消息列表中
    for msg in chat_history:
        # 检查 msg 中是否包含 'role' 键
        if not isinstance(msg, dict) or 'role' not in msg:
            print(f"警告：聊天历史消息格式错误，消息内容：{msg}")
            continue
        if msg["role"] == "user":
            # 添加用户消息
            messages.append(HumanMessage(content=msg["content"]))
        else:
            # 添加助手消息
            messages.append(AIMessage(content=msg["content"]))
    # 添加当前用户查询
    messages.append(HumanMessage(content=user_query))
    # 调用LLM获取回复
    response = llm.invoke(messages)
    return response.content

# 聊天历史保存，将聊天历史保存到JSON文件中
def save_history(session_id, chat_history):
    # 以写入模式打开JSON文件
    with open(f"history_{session_id}.json", "w", encoding="utf-8") as f:
        # 将聊天历史写入JSON文件
        json.dump(chat_history, f, ensure_ascii=False, indent=2)

# 聊天历史加载，从JSON文件中加载聊天历史
def load_history(session_id):
    try:
        # 以读取模式打开JSON文件
        with open(f"history_{session_id}.json", "r", encoding="utf-8") as f:
            # 从JSON文件中加载聊天历史
            chat_history = json.load(f)
    except FileNotFoundError:
        # 如果文件不存在，初始化为空列表
        chat_history = []
    return chat_history

if __name__ == "__main__":
    # 假设csv文件名为week14/table (1).csv
    books = load_books("week14/table (1).csv")
    # 获取用户输入的session_id，用于区分不同会话
    session_id = input("请输入session_id（用于区分不同会话）：").strip()
    # 加载聊天历史
    chat_history = load_history(session_id)
    print("欢迎来到图书馆咨询系统，输入'退出'结束。")
    while True:
        # 获取用户咨询输入
        user_input = input("你想咨询什么？")
        if user_input.strip() == "退出":
            break
        # 调用咨询函数获取回复
        answer = consult_book(books, user_input, chat_history)
        print("助手：", answer)
        # 将用户输入和助手回复添加到聊天历史
        chat_history.append({"role": "user", "content": user_input})
        chat_history.append({"role": "assistant", "content": answer})
        # 保存聊天历史
        save_history(session_id, chat_history)