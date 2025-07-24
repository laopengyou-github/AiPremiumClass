import os 
from dotenv import load_dotenv,find_dotenv
from openai import OpenAI

if __name__=='__main__':
    load_dotenv(find_dotenv())

    #创建调用客户端
    client = OpenAI(
        api_key=os.environ['API_KEY'],
        base_url=os.environ['BASE_URL']
    )
    response = client.chat.completions.create(
        model='glm-4-flash',
        messages=[
            {"role":"system","content":"你是一个善于聊天的AI助手"},
            {"role":"user","content":"你好,我是li老师！"},
            {"role":"assistant","content":"你好,li老师很高兴认识你！"},
            {"role":"user","content":"你如何称呼我"}],
            temperature=0.75,
            top_p=0.3,
            max_tokens=100
        )
    print(response.choices[0].message.content)