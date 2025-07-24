import os

from dotenv import load_dotenv, find_dotenv
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI

if __name__ == '__main__':
    load_dotenv(find_dotenv())

    books = [
        {"title": "百年孤独", "description": "马尔克斯的魔幻现实主义经典之作，讲述布恩迪亚家族七代人的故事。"},
        {"title": "三体", "description": "刘慈欣所著科幻小说，探讨人类文明与外星文明接触的可能性。"},
        {"title": "活着", "description": "余华代表作，通过一个普通农民的一生反映中国社会的巨大变迁。"}
    ]

    # 创建提示模板对象
    prompt_template = ChatPromptTemplate.from_messages([
        # ("system", "你是一个精通粤语的AI助理，还能把普通话翻译为粤语"),
        # ("system", "你是一个精通{language}的AI助理，还能把普通话翻译为{language}"),
        ("system", "你是一个图书管理系统的AI，能通过图书简介为借阅读者提供咨询,相关信息:{context}"),
        MessagesPlaceholder(variable_name="messages"),
    ])

    model = ChatOpenAI(
        model="glm-4-flash-250414",
        base_url=os.environ['base_url'],
        api_key=os.environ['api_key'],
        temperature=0.7
    )

    parser = StrOutputParser()

    # langchain表达式语言 (LCEL) 构建 chain
    chain = prompt_template | model | parser

    store = {}


    def get_session_hist(session_id):
        if session_id not in store:
            store[session_id] = InMemoryChatMessageHistory()
        return store[session_id]


    with_hist_run = RunnableWithMessageHistory(
        chain,
        get_session_history=get_session_hist,
        input_messages_key="messages"
    )
    session_id = "user_account_no"

    while True:
        text = input("请输入：")
        if text == "exit":
            break
        content = with_hist_run.invoke(
            {"context": books, "messages": [HumanMessage(content=text)]},
            config={"configurable": {"session_id": session_id}}
        )
        print('AI Message:', content)
