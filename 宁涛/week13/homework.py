import os
from dotenv import load_dotenv, find_dotenv
from openai import OpenAI

if __name__ == "__main__":
    load_dotenv(find_dotenv())
    client = OpenAI(api_key=os.environ['api_key'], 
                    base_url=os.environ['base_url'])
    
    response = client.chat.completions.create(
        model="glm-4-flash-250414", 
        messages=[
            {'role':'system', 'content':'你是一个擅长JAVA的AI助手'},
            {'role':'user', 'content':'请帮我写一个JAVA的冒泡排序'}
        ],
        temperature=0,
        max_tokens=500,
    )
    print(response.choices[0].message.content)
