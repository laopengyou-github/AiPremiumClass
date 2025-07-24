"""
通过langchain实现特定主题聊天系统，支持多轮对话。
"""
# 用于发现.env文件，并将其注册为环境变量
from dotenv import find_dotenv, load_dotenv
import os
# LLM调用封装
from langchain_openai import ChatOpenAI 
# 对话角色：user、assistant、system
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage 
# 输出结果解析
from langchain_core.output_parsers import StrOutputParser
# 对话历史
from langchain_community.chat_message_histories import FileChatMessageHistory, ChatMessageHistory

# 1. 导入必要包
from langchain_core.runnables.history import RunnableWithMessageHistory

# 导入template中传递聊天历史信息的“占位”类
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# 使用FileChatMessageHistory实现聊天历史的保存和加载

def get_session_history(session_id):
    # 文件名自动为 history_{session_id}.json
    # 通过sessionId 来获取不同的聊天记录
    # 加载对话历史
    return FileChatMessageHistory(f"history_{session_id}.json")

if __name__ == '__main__':
    # 加载环境变量
    load_dotenv(find_dotenv())

    # 初始化对话模型
    model = ChatOpenAI(
        # 模型名称
        model="glm-4-flash-250414",
        # 模型访问地址
        base_url=os.environ['ASE_URL'],
        # 模型访问key
        api_key=os.environ['API_KEY'],
        # 模型温度系数
        temperature=0.7
    )
 
    # 带有占位符的prompt template
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "你是一个得力的助手。你可以使用{lang}回答用户的问题。"),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )

    parser = StrOutputParser()
    # chain 构建
    chain =  prompt | model | parser

    # 在chain中注入聊天历史消息
    with_msg_hist = RunnableWithMessageHistory(
        chain, 
        get_session_history=get_session_history,
        input_messages_key="messages")
    
    # 动态输入session_id
    session_id = input('请输入session_id（用于区分不同会话）：').strip()

    while True:
        user_input = input('用户输入的Message (输入 "q" 退出)：')
        if user_input.lower() == 'q':
            break

        # 调用注入聊天历史的对象
        response = with_msg_hist.invoke(
            {
                # 用户输入的消息
                "messages":[HumanMessage(content=user_input)],
                "lang":"英语"
            },
            config={'configurable':{'session_id': session_id}})

        print('AI Message:', response)
