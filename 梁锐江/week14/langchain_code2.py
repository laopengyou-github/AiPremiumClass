import os
from dotenv import load_dotenv,find_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage,SystemMessage,AIMessage

if __name__ == '__main__':
    load_dotenv(find_dotenv())

    model = ChatOpenAI(
        model="glm-4-flash-250414",
        base_url=os.environ['base_url'],
        api_key=os.environ['api_key']
    )

    messages = [
        SystemMessage(content="你是一个翻译助手，你需要将输入的文本翻译成粤语。"),
        HumanMessage(content="请用粤语和我打一个热情的招呼！")
    ]

    # Invoke the model with a prompt
    response = model.invoke(messages)
    print(response.content)