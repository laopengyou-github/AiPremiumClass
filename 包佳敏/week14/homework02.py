from dotenv import find_dotenv, load_dotenv
from langchain_openai import ChatOpenAI #大模型调用封装LLM
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
import os

if __name__ == "__main__":
    # Load environment variables from .env file
    load_dotenv(find_dotenv())

    # Create a client for the OpenAI API
    model = ChatOpenAI(
        model="glm-4-flash-250414",
        openai_api_key=os.environ["api_key"],
        openai_api_base=os.environ["base_url"]
    )


    # template对象
    promt_temp= ChatPromptTemplate.from_messages([
        ("system", '你是一个只能图书管理的AI'),
        ("ai", '实现以下几个功能：'
            + '1、图书的借阅：提示用户如何操作，操作需扫图书二维码借阅'
            + '  （a)、检查用户有无登陆，未登陆提示用户登陆自己的图书账号'
            + '  （b)、提示用户如何操作，操作需扫图书二维码借阅'
            + '  （c）、显示图书封面、名字、作者、出版年份等'
            +'2、图书的归还'
            +'  （a)、检查用户有无登陆，未登陆提示用户登陆自己的图书账号'
            +'  （b)、提示用户如何操作，操作需扫图书二维码归还'
            +'  （c）、把扫码书籍放置还书架'
            +'3、为读者推荐图书：根据用户的年龄、性别、历史借阅记录分析、当下热门书籍等分析等为读者推荐书籍'
            + ' （a)、检查用户有无登陆，未登陆提示用户登陆自己的图书账号'
            +'  （b)、推荐结果分页显示，每页是10个'
            +'  （c）、显示图书封面、名字、作者、出版年份等'
        ),
        ("ai", '图书馆现有书目库存如下：'
            + '1、钢铁是怎样炼成的 10本'
            +'2、红楼梦 5本'
            +'3、小王子 7本'
            +'4、百年孤独 3本'
            +'5、活着 8本'
            +'6、平凡的世界 6本'
            +'7、围城 4本'
            +'8、三体 9本'
            +'9、时间简史 2本'
            +'10、追风筝的人 1本'
        ),
        ("user",'{text}')
    ])
    
    parser = StrOutputParser()
    #LLM结果解析器对象
    chain = promt_temp | model | parser

   #定制存储消息的dict
    store = {}

    # 定义函数：根据sessionId获取聊天记录（callback回调）
    def get_seesion_hist(session_id):
        if session_id not in store:
            store[session_id] = InMemoryChatMessageHistory() 
        return store[session_id] 

    #在chain中注入聊天历史消息
    with_msg_hist = RunnableWithMessageHistory(
        chain, 
        get_session_history=get_seesion_hist)#input_messages_key指明用户消息使用key
    session_id = "jiamin"
    
    while True:
        user_input = input("请问什么可以帮您：")
        if user_input.lower() == 'exit':
            break
        response = with_msg_hist.invoke(
            HumanMessage(content=user_input),
            config={'configurable':{'session_id':session_id}})
        print('AI Message:', response)
