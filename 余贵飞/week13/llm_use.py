from openai import OpenAI



if __name__ == '__main__':
    # 调用远程模型 key
    API_KEY = "b7cd18fd95214219beeb7a9478bf01f6.6Lz9lnGlgMQfB8OK"
    # 路径
    ASE_URL="https://open.bigmodel.cn/api/paas/v4"

    # 初始化chat
    client = OpenAI(
        api_key=API_KEY,
        base_url=ASE_URL
    )

    # 开始调用模型接口进行对话
    response = client.chat.completions.create(
        model="glm-4-flash-250414",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        stream=False
    )
    print(response.choices[0].message.content)

if __name__ == '__main__':
    while True:
        # 输入需要对话内容
        text = input("请输入对话内容：")
        # 将输入的对话内容与 prompt 拼接起来
        prompt = prompt + text
        # 将提示词提交大模型，进行对话
        sendRequest(prompt)




