import os
from dotenv import load_dotenv, find_dotenv
from openai import OpenAI

if __name__ == "__main__":
 
    load_dotenv(find_dotenv())
    
    # 创建调用客户端
    client = OpenAI(
        api_key=os.environ["api_key"],
        base_url=os.environ["base_url"]
    )

    # chat模式调用模型
    response = client.chat.completions.create(
        # 模型名称
        model="glm-4-plus",
        # 消息列表
        messages=[  # 聊天历史信息
            {'role':'system', 'content':'你是一个擅长与人聊天的AI助手'},
            {'role':'user', 'content':'你好,可以称呼我为马爸爸！'},
            {'role':'assistant', 'content':'你好马爸爸，很高兴认识你！'},
            {'role':'user', 'content':'我叫什么名字？'}
        ],
        # 模型参数
        temperature=0,
        # 最大token数
        max_tokens=500,
    )

    # 打印结果
    print(response.choices[0].message.content)