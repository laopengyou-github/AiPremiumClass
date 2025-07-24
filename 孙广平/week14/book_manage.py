import pandas as pd
from fastapi import FastAPI, Query
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langserve import add_routes
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import RunnableLambda
from typing import Dict, List
from pydantic import BaseModel  
import os

BOOKS_CSV = "books.csv"

def load_books():
    return pd.read_csv(BOOKS_CSV)

books_df = load_books()

# 构建文本知识库
def format_book_knowledge(df):
    return "\n\n".join(
        f"书名：《{row['title']}》\n作者：{row['author']}》\n分类：{row['category']}》\n简介：{row['description']}"
        for _, row in df.iterrows()
    )

book_knowledge = format_book_knowledge(books_df)

model = ChatOpenAI(
    model="deepseek-chat",
    base_url=os.environ['base_url'],
    api_key=os.environ['api_key'],
    temperature=0.7
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "你是图书馆的智能助手，负责为用户提供图书内容咨询、推荐、借书和还书等服务。\n\n以下是图书资料：\n{books}"),
    MessagesPlaceholder(variable_name="messages")
])

parser = StrOutputParser()

# 多轮对话链
store = {}
def get_session_hist(session_id):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

with_msg_hist_chain = RunnableWithMessageHistory(
    prompt | model | parser,
    get_session_history=get_session_hist,
    input_messages_key="messages"
)

borrowed_books: Dict[str, List[str]] = {}

def borrow_book(user_id: str, book_id: str):
    borrowed_books.setdefault(user_id, [])
    if book_id in borrowed_books[user_id]:
        return f"用户 {user_id} 已借阅图书 {book_id}"
    borrowed_books[user_id].append(book_id)
    return f"用户 {user_id} 成功借阅图书 {book_id}"

def return_book(user_id: str, book_id: str):
    if user_id not in borrowed_books or book_id not in borrowed_books[user_id]:
        return f"用户 {user_id} 未借阅图书 {book_id}"
    borrowed_books[user_id].remove(book_id)
    return f"用户 {user_id} 成功归还图书 {book_id}"

app = FastAPI()

class QAInput(BaseModel):
    question: str
    session_id: str

def input_adapter(inputs):
    question = inputs.get("question", "")
    return {
        "books": book_knowledge,
        "messages": [HumanMessage(content=question)]
    }

qa_chain_runnable = RunnableLambda(input_adapter) | with_msg_hist_chain

add_routes(app, qa_chain_runnable, path="/qa", input_type=QAInput)

@app.get("/books")
def list_books(keyword: str = Query(None, description="关键字，如标题、作者、分类")):
    if keyword:
        result = books_df[
            books_df.apply(lambda row: keyword.lower() in row.to_string().lower(), axis=1)
        ]
    else:
        result = books_df
    return result.to_dict(orient="records")

@app.get("/borrow")
def borrow(user_id: str, book_id: str):
    return {"message": borrow_book(user_id, book_id)}

@app.get("/return")
def return_(user_id: str, book_id: str):
    return {"message": return_book(user_id, book_id)}

# uvicorn book_manage:app --reload
# http://127.0.0.1:8000/qa/playground
