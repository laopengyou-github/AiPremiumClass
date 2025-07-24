import json
import re
import requests


if __name__ == "__main__":
    with open('./邪王真眼/week13/books.json', 'r', encoding='utf-8') as f:
        books = json.load(f)

    url = "http://localhost:11434/api/generate"

    while True:
        user_input = input("人, 说话: ")

        data = {
            "model":"deepseek-r1:8b",
            "prompt": f'''
                你是一个图书管理员，理解用户的需求，生成正确的回复，返回json格式，例子：
                {{
                    response：""已为您还书"",
                    cmd:""还明朝那些事儿""
                }}
                其中response是你要对用户说的，不管你要说设么都要房子啊这里，而不是json外面。
                cmd是执行结束还书操作的命令，如果不需要执行命令，cmd为空字符串。
                cmd中，第一个字“还”代表还书，“借”代表借书，后面接书名。
                当前书目：{books}。
                用户输入：{user_input}。
                用户问了图书相关的问题你要回答，如果用户没有具体需求或问题，你就问用户要办什么业务。
                注意一定要把回复放到json的response里面,不要有任何思考或多余的回答内容，把json放在代码块里面，用三个反引号包裹的那种。
               ''',
            "stream": False,
            "options": {
                "temperature": 0.3,
                "repeat_penalty": 1.0
            },
            "think": False,
        }

        response= requests.post(url, json=data)

        if response.status_code == 200:
            pattern = r'```json\n([\s\S]*?)\n```'
            match = re.search(pattern, json.loads(response.text)['response'])

            if match:
                data = json.loads(match.group(1))
            else:
                print('重试')
                continue

            response = data['response']
            cmd = data['cmd']
            if len(cmd) > 0:
                if cmd[0] == '还':
                    for book in books:
                        if book['title'] == cmd[1:]:
                            book['quantity'] += 1
                            break
                    print(f"系统, 说话: {cmd[1:]}还了一本，剩余{book['quantity']}")
                elif cmd[0] == '借':
                    for book in books:
                        if book['title'] == cmd[1:]:
                            book['quantity'] -= 1
                            break
                    print(f"系统, 说话: {cmd[1:]}被借走了一本，剩余{book['quantity']}")
            
            print("AI, 说话: " + response)