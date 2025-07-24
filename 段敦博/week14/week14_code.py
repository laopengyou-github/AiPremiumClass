#1.通过langchain实现特定主题聊天系统，支持多轮对话


from dotenv import find_dotenv,load_dotenv
import 
import json
from langchain_openai import ChatOpenAI #LLM调用封装
from langchain_core.messages import HumanMessage,AIMessage,SystemMessage #对话角色:user,assistant,system
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import BaseChatMessageHistory,InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnabeWithMessageHistory

#导入template中传递聊天历史信息的“占位”类
from langchain_core.prompts import ChatPromptTemplate,MessagesPlceholder

def save_history(session_id):
    #Message 对象中to_json() 消息转换json格式
    history =[msg.to_json() for msg in store[session_id].messages]
    with open(f'history_{session_id}.json',"w",encoding="utf-8") as f:
        json.dump(history,f,ensure_ascii =False,indent=2)


#实现：聊天历史保存（不同session_id）
def load_history(session_id):
#加载失败情况下，跳转到except(错误)
    try:
        with open(f"history_{session_id}.json","r",encoding="utf-8") as f:
            messages=json.load(f)
        history=ChatMessageHistory()
        history.parse_json(messages)
        store[session_id]=history
    except FileNotFoundError:
        store[session_id] =ChatMessageHistory()


if __name__ =='__main__':
    load_dotenv(find_dotenv())

     #Initialize the OpenAI Chat model
    model=ChatOpenAI(
        model="glm-4-flash-250414",
        base_url=os.environ['base_url'],
        api_key=os.environ['api_key'],
        temperature =0.7
    )

    #带有占位符的prompt template
    prompt =ChatPromptTemplate.from_messages(
        [
            ("system","你是一个得力的助手。尽你所能使用{lang}回答所有问题"),
            MessagesPlaceholder(variable_name="messages")
        ]
    )
    
    parser=StrOutputParser()

    #chain构建
    chain=prompt | model | parser

    #定制存储消息的dict
    store ={}


    #定义函数：根据sessionId获取聊天记录(callback 回调)
    def get_session_hist(session_id):
        #以sessionid为key从store中提取关联聊天历史对象
        if session_id not in store:
            laod_history(session_id)
        return store[session_id] #检索 （langchain负责维护内部聊天历史信息）


    #在chain中注入聊天历史消息
    with_msg_hist=RunnabeWithMessageHistory(chain,
                                            get_session_history=get_session_hist,
                                            input_messages_key="essages") #input_messages_key指明用户消息使用key

    #动态输入session_id
    session_id =input('请输入session_id(用于区分不同绘画)：').strip()

    while True:
        user_input =input('用户输入的Message：')
        if user_input.lower() =='quit':
            break

        #调用注入聊天历史的对象
        response=with_msg_hist.invoke(
            {
                "messages":[HumanMessage(content=user_input)],
                "lang":"汉语"
            },
            config ={'configurable':session_id})
            
        print('AI Message:', response)
        #聊天历史自动保存
        save_history(session_id)


#2.借助langchain实现图书管理系统开发扩展，通过图书简介为借阅读者提供咨询

#先让AI生成小说的信息

from dotenv import find_dotenv,load_dotenv
import os
import json
from langchain_openai import ChatOpenAI #LLM调用封装
from langchain_core.messages import HumanMessage,AIMessage,SystemMessage #对话角色:user,assistant,system
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import BaseChatMessageHistory,InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnabeWithMessageHistory
#导入template中传递聊天历史信息的“占位”类
from langchain_core.prompts import ChatPromptTemplate,MessagesPlceholder


#加载.env文件
load_dotenv()
API_KEY=os.getenv("api_key")
BASE_URL=os.getenv("base_url")


#初始化LLM
llm=ChatOpenAI(
    openai_api_key=API_KEY,
    openai_api_base=BASE_URL,
    model_name="glm-4-flash", #可根据实际情况更换
    temperature =0.7
)

#读取图书信息
def load_books(csv_path):
    df=pd.read_csv(csv_path)
    books =df.to_dict(orient="records")
    return books

#生成图书简介（如简介为空时自动生成）
def generate_book_intro(book):
    if pd.isna(book["简介"]) or not book["简介"]:
        prompt=f"请为以下图书生成简短简介：书名 {book['书名']}.作者:{book['作者']}"
        response=llm.invoke([HumanMessage(content=prompt)])
    return book["简介"]


#图书咨询对话
def consult_book(books,user_query,chat_history=None):
    if chat_history is None:
        chat_history=[]
    #拼接所有图书信息，去掉库存字段
    books_info ="\n".join([
        f"{b['编号']}{b['书名']}，作者:{b['作者']}，类型：{b['类型']}，简介：{generate_book_intro}"
        for b in books
    ])
    system_prompt =(
        "你是一个图书馆咨询助手，以下是图书信息:\n"
        f"{books_info}\n"
        "请根据用户提问，结合图书简介和聊天历史，给出专业，简明的推荐或解答"
    )
    messages =[SystemMessage(content=system_prompt)]
    for msg in chat_history:
        if msg["role"] =="user":
            messages.append(HumanMessage(content=msg["content"]))
        else:
            messages.append(AIMessage(content=msg["content"]))
    messages.append(HumanMessage(content=user_query))
    response =llm.invoke(messages)
    return repsonse.content

#聊天历史保存
def save_history(session_id,chat_history):
    with open(f"history_{session_id}.json","w",encoding="utf-8") as f:
        json.dump(chat_history,f,ensure_ascii=False,indent=2)

#聊天历史加载
def load_history(session_id):
    try:
        with open(f"history_{session_id}.json","r",encoding="utf-8") as f:
            messages=json.load(f)
        history=ChatMessageHistory()
        history.parse_json(messages)
        store[session_id]=history
    except FileNotFoundError:
        store[session_id] =ChatMessageHistory()


if __name__ =='__main__':
    #假设csv文件名为books.csv
    books=load_books("books.csv")
    session_id =input('请输入session_id(用于区分不同绘画)：').strip()
    chat_history=load_history(session_id)
    print("欢迎来到图书馆查询系统，输入';退出'结束")
    while True:
        user_input =input('用户输入的Message：')
        if user_input.lower() =='quit':
            break
        answer=consult_book(books,user_input,chat_history)
        print("助手："answer)
        chat_history.append({"role":"user","content":user_input})
        chat_history.append({"role":"assistant","content":answer()})
        save_history(session_id,chat_history)






