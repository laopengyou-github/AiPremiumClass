from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from dotenv import load_dotenv
import os
from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.tools.retriever import create_retriever_tool
from langchain_community.vectorstores import FAISS
from langchain.agents import create_tool_calling_agent, AgentExecutor

if __name__ == '__main__':
    load_dotenv()

    llm = ChatOpenAI(
        model='o3-mini-2025-01-31',
        api_key=os.environ['api_key'],
        base_url=os.environ['base_url']
    )

    embeddings = OpenAIEmbeddings(
        model='text-embedding-3-small',
        api_key=os.environ['api_key'],
        base_url=os.environ['base_url']
    )

    prompt = hub.pull('hwchase17/openai-functions-agent')

    if not os.path.exists('save_local'):
        docs = WebBaseLoader('https://hawstein.com/2013/03/26/dp-novice-to-advanced/').load()
        splitter = RecursiveCharacterTextSplitter(
            separators=['\n\n', '\n', ''],
            chunk_size=1000,
            chunk_overlap=100,
        )

        splited_docs = splitter.split_documents(docs)

        vector_store = FAISS.from_documents(splited_docs, embeddings)
        vector_store.save_local('save_local')
    else:
        vector_store = FAISS.load_local('save_local', embeddings, allow_dangerous_deserialization=True)
    
    retriever = vector_store.as_retriever(search_kwargs={'k': 2})
    retriever_tool = create_retriever_tool(retriever,
                                        'DynamicProgrammingGuide',  # 修改后的工具名称
                                        '动态规划算法通常基于一个递推公式及一个或多个初始状态。 当前子问题的解将由上一次子问题的解推出。使用动态规划来解题只需要多项式时间复杂度， 因此它比回溯法、暴力法等要快许多。')

    tavily_tool = TavilySearchResults(max_results=2)

    tools = [retriever_tool, tavily_tool]

    agent = create_tool_calling_agent(
        llm=llm,
        tools=tools,
        prompt=prompt
    )
    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
    )

    response = executor.invoke(
        {
            'input': '什么是动态规划？'
        }
    )

    print(response['output'])
