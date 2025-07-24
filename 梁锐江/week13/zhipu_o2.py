import os
from dotenv import load_dotenv, find_dotenv
from openai import OpenAI

if __name__ == '__main__':
    load_dotenv(find_dotenv())

    client = OpenAI(
        api_key=os.getenv("api_key"),
        base_url=os.getenv("base_url")
    )

    response = client.chat.completions.create(
        model="glm-z1-flash",
        messages=[
            {"role": "system", "content": "你是一个智能图书管理AI,能够实现图书借阅和归还,并且能根据喜好为读者推荐图书"},
            {"role": "user", "content": "我想要还书，并且借一本了解中国历代历史的书"}
        ],
        stream=False,
        temperature=0,  # 全局调整所有词的概率分布。
        top_p=1,  # 局部截断，直接排除低概率词。

    )

    print(response.choices[0].message.content)
