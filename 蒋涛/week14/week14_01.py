"""
通过langchain实现特定主题聊天系统，支持多轮对话。
"""

from dotenv import find_dotenv, load_dotenv
import os
from langchain_openai import ChatOpenAI # LLM调用封装
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage # 对话角色：user、assistant、system
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_message_histories import FileChatMessageHistory, ChatMessageHistory

# 1. 导入必要包
from langchain_core.runnables.history import RunnableWithMessageHistory

# 导入template中传递聊天历史信息的“占位”类
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# 使用FileChatMessageHistory实现聊天历史的保存和加载

def get_session_history(session_id):
    """
    根据会话ID获取聊天历史记录对象。

    :param session_id: 会话的唯一标识符
    :return: FileChatMessageHistory对象，用于保存和加载聊天历史
    """
    # 文件名自动为 history_{session_id}.json
    return FileChatMessageHistory(f"history_{session_id}.json")

if __name__ == '__main__':
    # 加载环境变量
    load_dotenv(find_dotenv())

    # Initialize the OpenAI Chat model
    model = ChatOpenAI(
        model="glm-4-flash-250414",
        base_url=os.environ['BASE_URL'],
        api_key=os.environ['API_KEY'],
        temperature=0.7
    )
 
    # 带有占位符的prompt template
    prompt = ChatPromptTemplate.from_messages(
        [
            # 系统消息，设定助手的角色和回答语言
            ("system", "你是一个得力的助手。你可以使用{lang}回答用户的问题。"),
            # 聊天消息占位符，用于插入历史聊天记录
            MessagesPlaceholder(variable_name="messages"),
        ]
    )

    # 初始化输出解析器，将模型输出转换为字符串
    parser = StrOutputParser()
    # chain 构建，将prompt、模型和解析器按顺序连接
    chain =  prompt | model | parser

    # 在chain中注入聊天历史消息
    with_msg_hist = RunnableWithMessageHistory(
        chain, 
        get_session_history=get_session_history,
        input_messages_key="messages"
    )
    
    # 动态输入session_id，用于区分不同的会话
    session_id = input('请输入session_id（用于区分不同会话）：').strip()

    while True:
        # 获取用户输入
        user_input = input('用户输入的Message (输入 "quit" 退出)：')
        # 若用户输入 "quit"，则退出循环
        if user_input.lower() == 'quit':
            break

        # 调用注入聊天历史的对象
        response = with_msg_hist.invoke(
            {
                "messages":[HumanMessage(content=user_input)],
                "lang":"汉语"
            },
            config={'configurable':{'session_id': session_id}}
        )

        # 打印AI的回复
        print('AI Message:', response)
