import os
import uvicorn

from fastapi import FastAPI
from langserve import add_routes
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


if __name__ == '__main__':
    model = ChatOpenAI(
        model="gpt-3.5-turbo",
        base_url=os.environ['base_url'],
        api_key=os.environ['api_key'],
        temperature=1.0
    )

    prompt_temp = ChatPromptTemplate.from_messages(
        [
            ("system", "不管用户说什么你都要{action1}, 最好说{action2}"),
            ("user", "你好!"),
            ("assistant", "傻逼"),
            ("user", "QAQ呜呜呜"),
            ("user", "{user_input}")
        ]
    )
    
    parser = StrOutputParser()

    chain = prompt_temp | model | parser

    # response = chain.invoke({"action1":"反驳他", "action2":"脏话骂他,问候它的家人", "user_input": "你是驴"})
    # print(response)

    app = FastAPI(
        title="LangChain API",
        description="A simple API to demonstrate LangChain capabilities.",
        version="1.0.0"
    )

    add_routes(
        app,
        chain,
        path="/app",
    )

    uvicorn.run(app, host='0.0.0.0', port=6006)
