from dotenv import find_dotenv,load_dotenv
import os
from langchain_openai import ChatOpenAI # LLM调用封装
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage # 对话角色：user、assistant、system
from langchain_core.output_parsers import StrOutputParser  # 解析器设置
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory # 聊天历史
from langchain_core.runnables.history import RunnableWithMessageHistory # 注入聊天历史
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder # prompt模板，占位类

if __name__ == '__main__':
    try:
        # 1、加载环境变量
        load_dotenv(find_dotenv())
        # 2、初始化OpenAI Chat模型
        model = ChatOpenAI(
            model='glm-4-flash-250414',
            base_url=os.environ['base_url'],
            api_key=os.environ['api_key'],
            temperature = 0.75
        )
        # 3、定义prompt模板
        prompt = ChatPromptTemplate.from_messages(
            [
                # ('system','你是一个非常幽默的AI助手，用{lang}回答所有问题.'),
                ('system','你是一个图书馆的AI管家，根据用户想查询图书的简介进行回答问题，你可以自己构建一个小的图书馆让用户借阅或者还书用，用{lang}回答所有问题.'),
                ('user','你知道《西游记》这本书吗?'),
                ('assistant','《西游记》是中国神魔小说的经典之作,该小说主要讲述了孙悟空出世，并寻菩提祖师学艺及大闹天宫后，与猪八戒、沙僧和白龙马一同护送唐僧西天取经，于路上历经险阻，降妖除魔，渡过了九九八十一难，成功到达大雷音寺，向如来佛祖求得《三藏真经》，最后五圣成真的故事。该小说以“玄奘取经”这一历史事件为蓝本，经作者的艺术加工，深刻地描绘出明朝时期的社会生活状况。'),
                ('user','我要借这一本书'),
                ('assistant','这是一本非常好的书，当前书库还有5本，借给你一本就还剩4本，记得一周内归还或者延期借阅。'),
                MessagesPlaceholder(variable_name='messages')
            ]
        )
        # 4、定义解析器
        parser = StrOutputParser()
        # 5、构建chain
        chain = prompt | model | parser
        # 6、定制存储消息的dict,里面存放的key和value
        # key:  sessionId会话Id（资源编号）区分不同用户或不同聊天内容
        # value: InMemoryChatMessageHistory存储聊天信息list对象
        store = {}
        # callback回调函数定义，根据sessionId获取聊天历史 （callback）
        def get_session_history(session_id):
            if session_id not in store:
                store[session_id] = InMemoryChatMessageHistory()  # 创建消息存储的内存，适用临时简短会话
            
            return store[session_id]      # 检索 （langchain负责维护内部聊天历史信息）
            
        # 7、将chain注入聊天历史
        # 7.1调用chain之前，还需要根据sessionID提取不同的聊天历史
        with_msg_hist = RunnableWithMessageHistory(
            chain,
            get_session_history=get_session_history,
            input_messages_key='messages'
        )
        # 7.2 连续对话功能(循环调用)
        session_id = 'abc123'
        while True:
            # 用户输入
            user_input = input('用户输入的Message：')
            [ exit(0) if user_input=='exit' else None ]
            #断言
            assert isinstance(user_input, str)
            # 调用注入聊天历史的对象
            response = with_msg_hist.invoke(
                {
                    'messages':[HumanMessage(content=user_input)],
                    'lang':'四川话'
                },
                config = {'configurable':{'session_id': session_id}}
            )
            print('AI Message:', response)
    except Exception as e:
        print(e)