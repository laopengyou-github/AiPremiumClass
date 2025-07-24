import os
from dotenv import load_dotenv, find_dotenv
from openai import OpenAI

if __name__ == '__main__':
    load_dotenv(find_dotenv())

    client = OpenAI(
        api_key=os.getenv("api_key"),
        base_url=os.getenv("base_url")
    )

    # 初始化对话历史，包含系统角色设定
    messages = [
        {"role": "system", "content": "你是一个智能图书管理AI,能够实现图书借阅和归还,并且能根据喜好为读者推荐图书"}
    ]

    while True:
        # 1. 获取用户输入
        user_input = input("用户：")
        if user_input.lower() in ["退出", "exit", "q"]:
            break

        messages.append({"role": "user", "content": user_input})

        response = client.chat.completions.create(
            model="glm-z1-flash",
            messages=messages,
            stream=False,
            temperature=0,  # 全局调整所有词的概率分布。
            top_p=1,  # 局部截断，直接排除低概率词。

        )

        # 4. 获取AI回复并打印
        ai_response = response.choices[0].message.content
        print(f"AI: {ai_response}")

        # 5. 将AI回复添加到对话历史
        messages.append({"role": "assistant", "content": ai_response})
