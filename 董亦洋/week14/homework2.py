from dotenv import find_dotenv, load_dotenv
import os
from langchain_openai import ChatOpenAI # LLM调用封装
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage # 对话角色：user、assistant、system
from langchain_core.output_parsers import StrOutputParser

# 1. 导入必要包
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# 导入template中传递聊天历史信息的“占位”类
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


if __name__ == '__main__':
    load_dotenv(find_dotenv())

    # Initialize the OpenAI Chat model
    model = ChatOpenAI(
        model="glm-4-flash-250414",
        base_url=os.environ['base_url'],
        api_key=os.environ['api_key'],
        temperature=0.7
    )
 
    # 带有占位符的prompt template
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "你是一个专业的图书管理系统，你有如下四个功能：\
             1.借阅图书功能：记录借阅人ID/姓名、借阅日期、预计归还日期(默认30天)；检查图书库存状态；更新图书可借数量；处理并发借阅请求；包含输入验证(如借阅人信息有效性、图书是否存在等)；返回借阅成功/失败信息及原因。\
             2.归还图书功能：验证归还图书信息；记录实际归还日期；计算是否逾期及相应罚金；更新图书库存状态；处理图书损坏/丢失等特殊情况；提供归还确认信息。\
             3.查询图书：条件图书查询系统，支持：按书名(支持模糊匹配)、作者、ISBN、出版社、分类等多字段组合查询。\
             4.增加库存：新书信息录入；已有图书数量增加；自动生成唯一图书ID；支持批量导入；记录入库时间和操作人员；提供操作确认反馈。"),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )

    parser = StrOutputParser()
    # chain 构建
    chain =  prompt | model | parser
    
    # 定制存储消息的dict
    # key:  sessionId会话Id（资源编号）区分不同用户或不同聊天内容
    # value: InMemoryChatMessageHistory存储聊天信息list对象
    store = {}

    # 定义函数：根据sessionId获取聊天历史（callback 回调）
    # callback 系统调用时被执行代码
    def get_session_hist(session_id):
        # 以sessionid为key从store中提取关联聊天历史对象
        if session_id not in store:
            store[session_id] = InMemoryChatMessageHistory() # 创建
        return store[session_id] # 检索 （langchain负责维护内部聊天历史信息）

    # 在chain中注入聊天历史消息
    # 调用chain之前，还需要根据sessionID提取不同的聊天历史
    with_msg_hist = RunnableWithMessageHistory(
        chain, 
        get_session_history=get_session_hist,
        input_messages_key="messages")  # input_messages_key 指明用户消息使用key
    
    # session_id
    session_id = "abc123"

    while True:
        # 用户输入
        user_input = input('用户输入的Message：')
        # 调用注入聊天历史的对象
        response = with_msg_hist.invoke(
            {
                "messages":[HumanMessage(content=user_input)]
            },
            config={'configurable':{'session_id': session_id}})

        print('AI Message:', response)

