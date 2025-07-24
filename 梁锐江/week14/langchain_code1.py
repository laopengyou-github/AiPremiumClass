import os
from dotenv import load_dotenv,find_dotenv
from langchain_openai import ChatOpenAI

if __name__ == '__main__':
    load_dotenv(find_dotenv())

    model = ChatOpenAI(
        model="glm-4-flash-250414",
        base_url=os.environ['base_url'],
        api_key=os.environ['api_key']
    )
    # Invoke the model with a prompt
    response = model.invoke('请用粤语和我打一个热情的招呼！')
    print(response.content)