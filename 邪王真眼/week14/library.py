import os
import json

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory


if __name__ == '__main__':
    with open('./邪王真眼/week13/books.json', 'r', encoding='utf-8') as f:
        books = json.load(f)

    model = ChatOpenAI(
        model="gpt-3.5-turbo",
        base_url=os.environ['base_url'],
        api_key=os.environ['api_key'],
        temperature=0.7
    )

    prompt_temp = ChatPromptTemplate.from_messages(
        [
            MessagesPlaceholder(variable_name="messages"),
            ("system", '''你是一个图书管理员, 返回json格式。
             例子：
                {{
                    response:""已为您还书"",
                    cmd:""还明朝那些事儿""
                }}
                其中response是你要对用户说的,不管你要说什么都这里,而不是json外面。
                cmd是执行结束还书操作的命令,如果不需要执行命令,cmd为空字符串。
                cmd中,第一个字“还”代表还书,“借”代表借书,后面接书名。
                当前书目：{books}。
                这里的数目是实时更新的，你不用计算用户借走了几本，以这里的数据为准。一个人同一本书可以借多本，没有任何限制,数量为0就不借。
                用户问了图书相关的问题你要回答,可以通过图书简介为借阅读者提供咨询,如果用户没有具体需求或问题,你就跟用户聊天。
                注意一定要把回复放到json的response里面,不要有任何多余的字符。不要把json放在代码块里面,不要返回纯文本,会解析json失败的!
             '''),
        ]
    )
    
    parser = StrOutputParser()

    chain = prompt_temp | model | parser

    store = {}

    def get_session_hist(session_id):
        if session_id not in store:
            store[session_id] = InMemoryChatMessageHistory()
        return store[session_id]

    with_msg_hist = RunnableWithMessageHistory(
        chain, 
        get_session_history=get_session_hist,
        input_messages_key="messages")

    while True:
        user_input = input('你，说话: ')

        response = with_msg_hist.invoke(
            {
                "messages":[HumanMessage(content=user_input)],
                "books":f"{books}",
            },
            config={'configurable':{'session_id': "abc123"}}
            )
        
        try:
            data = json.loads(response)

            res = data['response']
            cmd = data['cmd']
            if len(cmd) > 0:
                if cmd[0] == '还':
                    for book in books:
                        if book['title'] == cmd[1:]:
                            book['quantity'] += 1
                            break
                    print(f"系统, 说话: {cmd[1:]}还了1本，剩余{book['quantity']}本")
                elif cmd[0] == '借':
                    for book in books:
                        if book['title'] == cmd[1:]:
                            book['quantity'] -= 1
                            break
                    print(f"系统, 说话: {cmd[1:]}被借走了1本，剩余{book['quantity']}本")
            
            print("AI, 说话: " + res)
        except:
            print("AI, 说话: " + response)
            continue
