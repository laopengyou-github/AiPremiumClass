import os
from dotenv import load_dotenv, find_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from fastapi import FastAPI
from langserve import add_routes
import uvicorn

if __name__ == '__main__':
    load_dotenv(find_dotenv())

    # 创建提示模板对象
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", "你是一个精通{language}的AI助理，还能把普通话翻译为{language}"),
        ("user", "{text}")
    ])

    model = ChatOpenAI(
        model="glm-4-flash-250414",
        base_url=os.environ['base_url'],
        api_key=os.environ['api_key'],
        temperature=0.7
    )

    parser = StrOutputParser()

    chain = prompt_template | model | parser

    content = chain.invoke({"language": "粤语", "text": "请用粤语和我打一个热情的招呼！"})

    # FastAPI
    app = FastAPI(
        title="LangChain API",
        description="A simple API to demonstrate LangChain capabilities.",
        version="1.0.0"
    )

    add_routes(
        app,
        chain,
        path="/app_tranlate"
    )

    uvicorn.run(app, host='localhost', port=6006)

