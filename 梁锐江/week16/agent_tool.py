from dotenv import load_dotenv,find_dotenv
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain import hub
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_community.tools import TavilySearchResults
from langchain_core.tools import create_retriever_tool
from langchain_openai import ChatOpenAI
import os
from langchain_community.vectorstores import FAISS
from langchain_tavily import TavilySearch

if __name__ == '__main__':
    load_dotenv(find_dotenv())
    # print("api_key:", os.environ['api_key'])
    # print("base_url:", os.environ['base_url'])


    model = ChatOpenAI(
        model="glm-4-flash-250414",
        api_key=os.environ['api_key'],
        base_url=os.environ['base_url']
    )

    prompt = hub.pull("hwchase17/openai-functions-agent")

    # 大模型的联网搜索
    search = TavilySearch(max_result=2)

    embedding_model = OpenAIEmbeddings(
        model="embedding-3",
        api_key=os.environ['api_key'],
        base_url=os.environ['base_url']
    )

    vector_store = FAISS.load_local("local_save",
                                    embeddings=embedding_model,
                                    allow_dangerous_deserialization=True)

    retriever = vector_store.as_retriever(search_kwargs={"k": 2})
    retriever_tool = create_retriever_tool(
        retriever,
        "book_retriever",
        description="围绕《把时间当作朋友》核心内容，从心智力量、时间管理、自律坚持、兴趣与努力的关系、对成功学的反思等多方面，阐述了通过开启和管理心智、合理规划时间、克服懒惰、持续行动等实现个人成长与改变的理念。"
    )

    tools = [search, retriever_tool]

    agent = create_tool_calling_agent(
        llm=model,
        prompt=prompt,
        tools=tools,
    )

    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
    )

    msgs = executor.invoke({"input": "围绕《把时间当作朋友》，说明如何做好时间管理"})

    print(msgs['output'])
