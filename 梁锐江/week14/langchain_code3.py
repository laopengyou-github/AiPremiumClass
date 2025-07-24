import os
from dotenv import load_dotenv,find_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage,SystemMessage,AIMessage
from langchain_core.prompts import ChatPromptTemplate

if __name__ == '__main__':
    # 创建提示模板对象
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", "你是一个精通{language}的AI助理，还能把普通话翻译为{language}"),
        ("user", "{text}")
    ])

    # 调用模板对象
    resp = prompt_template.invoke({"language": "粤语", "text": "请用粤语和我打一个热情的招呼！"})
    print(resp)
