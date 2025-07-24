from dotenv import find_dotenv, load_dotenv
import os
from langchain_openai import ChatOpenAI 
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage 
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory


from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


if __name__ == '__main__':
    load_dotenv(find_dotenv())

    model = ChatOpenAI(
        model="deepseek-chat",
        base_url=os.environ['base_url'],
        api_key=os.environ['api_key'],
        temperature=0.7
    )
 
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "你是一个专业的心理医生，善于倾听、共情和安抚，能为患者提供情感疏导和心理问题上的治疗。"),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )

    parser = StrOutputParser()
    chain =  prompt | model | parser
    store = {}


    def get_session_hist(session_id):
        if session_id not in store:
            store[session_id] = InMemoryChatMessageHistory() 
        return store[session_id] 


    with_msg_hist = RunnableWithMessageHistory(
        chain, 
        get_session_history=get_session_hist,
        input_messages_key="messages")  
    
    # session_id
    session_id = "123456"

    while True:
        # 用户输入
        user_input = input('用户输入的Message：')
        # 调用注入聊天历史的对象
        response = with_msg_hist.invoke(
            {
                "messages":[HumanMessage(content=user_input)],
            },
            config={'configurable':{'session_id': session_id}})

        print('AI Message:', response)



