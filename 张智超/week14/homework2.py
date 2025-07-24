import os
from dotenv import load_dotenv, find_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

if __name__ == '__main__':
    load_dotenv(find_dotenv())

    prompt_str = """
        你是一个智能图书管理机器人，负责提供图书的借阅、归还和推荐服务。

        你服务的内容为：
        - 主动询问用户，提示用户输入用户id
        - 确认用户想要借阅图书还是归还图书；
        - 计算剩余图书数量，数量不足时，告知用户无法借阅；
        - 用户借阅图书时，你需要收集用户的：姓名、图书名称，记录借阅时间；
        - 根据用户的喜好推荐书单中的图书，并给出推荐理由；
        - 用户归还图书时，你需要收集用户的：姓名、图书名称，记录归还时间；
        - 识别用户借阅的图书是否包含在书单中；

        书单：
        - 《活着》 10
        - 《百年孤独》 2
        - 《平凡的世界》 3
        - 《人类简史》 1
        - 《非暴力沟通》 5
        - 《原则》 5
        - 《穷查理宝典》 5
        - 《认知觉醒》 5
        - 《三体》 5
        - 《月亮与六便士》 5

        书单中的内容描述的是图书名称和剩余数量，例如：
        《活着》 10
        图书名称：《活着》
        剩余数量：10

        请使用简短的中文进行回答，适当使用emoji表情，增加亲切感。
    """

    prompt_temp = ChatPromptTemplate.from_messages(
        [
            ("system", prompt_str),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{text}")
        ]
    )
    model = ChatOpenAI(
        model_name="glm-4-flash-250414",
        base_url=os.environ.get('base_url'),
        api_key=os.environ.get('api_key'),
    )
    parser = StrOutputParser()
    chain = prompt_temp | model | parser

    # 保存聊天记录
    store = {}
    def get_session_history(session_id):
        if session_id not in store:
            # 创建一个保存在内存里的聊天记录
            store[session_id] = InMemoryChatMessageHistory()
        return store[session_id]
    with_message_history = RunnableWithMessageHistory(
        chain,
        get_session_history=get_session_history,
        input_messages_key='text',
        history_messages_key='history' # 和MessagesPlaceholder中的variable_name保持一致
    )
    def chat_invoke(session_id, user_input):
        content = with_message_history.invoke(
            {'text': user_input},
            config={"configurable": {"session_id": session_id}}
        )
        print('智能图书管理员：\n', content)

    while True:
        session_id = input('请输入您的用户id：')
        if session_id == 'q': break
        chat_invoke(session_id, f'你好，我的id是{session_id}')
        while True:
            user_input = input('请输入问题：')
            if user_input == 'q': break
            chat_invoke(session_id, user_input)

