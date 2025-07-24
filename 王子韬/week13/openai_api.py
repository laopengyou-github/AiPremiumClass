import os
from openai import OpenAI
from dotenv import find_dotenv, load_dotenv


def call_gpt(temperature=1.0, top_p=1.0, max_tokens=100):
    load_dotenv(find_dotenv())
    client = OpenAI(
        api_key=os.getenv("API_KEY"),
        base_url=os.getenv("BASE_URL"),
    )

    responses = client.chat.completions.create(
        model = 'glm-4-flash-250414',
        messages=[{"role": "user", "content": "你有什么优势，限制50字"}],
        temperature = temperature,
        top_p = top_p,
        max_tokens = max_tokens,
    )

    return responses.choices[0].message.content

if __name__ == "__main__":

    # temperature 0.0~2.0
    # top_p 0.0~2.0
    # 稳定输出
    response = call_gpt(temperature=0.2, top_p=1.0)
    print(response)
    # 多样输出
    response = call_gpt(temperature=0.5, top_p=0.9)
    print(response)
    # 混乱输出
    response = call_gpt(temperature=1.0, top_p=0.5)
    print(response)
