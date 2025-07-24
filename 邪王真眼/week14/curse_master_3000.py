import os

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory


if __name__ == '__main__':
    model = ChatOpenAI(
        model="gpt-3.5-turbo",
        base_url=os.environ['base_url'],
        api_key=os.environ['api_key'],
        temperature=1.0
    )

    prompt_temp = ChatPromptTemplate.from_messages(
        [
            ("system", "不管用户说什么你都要{action1}, 最好说{action2}"),
            ("user", "你好!"),
            ("assistant", "傻逼"),
            ("user", "QAQ呜呜呜"),
            MessagesPlaceholder(variable_name="messages"),
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
        user_input = input('用户输入的Message：')

        for r in with_msg_hist.stream(
            {
                "messages":[HumanMessage(content=user_input)],
                "action1":"反驳他",
                "action2":"脏话骂他,问候它的家人"
            },
            config={'configurable':{'session_id': "abc123"}}
            ):
            print(r, end='')
        print('\n')

