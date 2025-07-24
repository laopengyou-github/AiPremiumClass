import os 
from dotenv import load_dotenv,find_dotenv

from langchain_openai import ChatOpenAI,OpenAIEmbeddings
from langchain import hub
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import WebBaseLoader,PDFMinerLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.tools.retriever import create_retriever_tool
from langchain.agents import create_tool_calling_agent, AgentExecutor



if __name__ == "__main__":

    #加载环境变量
    load_dotenv(find_dotenv())
    print(f"TAVILY_API_KEY: {'已设置' if os.environ.get('TAVILY_API_KEY') else '未设置'}")

    #加载模型
    model = ChatOpenAI(
        model="glm-4-flash-250414",
        base_url=os.environ['base_url'],
        api_key=os.environ['api_key']
    )

    #创建prompt
    prompt = hub.pull("hwchase17/openai-functions-agent")

    #创建搜索工具
    search = TavilySearchResults(max_results=2)
    # print(search.invoke("今天深圳天气如何"))

    #构建Embedding_model
    embedding_model = OpenAIEmbeddings(
        model="embedding-3",
        api_key=os.environ["embedding_api_key"],
        base_url=os.environ["base_url"]
    )

    if not os.path.exists("agent_save_local"):
        # 如果本地没有保存的向量数据库，则创建新的向量数据库
        print('开始创建新的向量数据库...')

        loader = WebBaseLoader(web_path="https://hawstein.com/2014/03/06/make-thiner-make-friend-with-time/")
        docs = loader.load()

        #拆分句子
        splitter = RecursiveCharacterTextSplitter(
            separators=['\n\n', '\n', ''],
            chunk_size=1000,
            chunk_overlap=100
        )
        splited_docs = splitter.split_documents(docs)

        #构建向量数据库
        vector_store = FAISS.from_documents(
            documents=splited_docs,
            embedding=embedding_model,
        )
        vector_store.save_local("agent_save_local")
        print('faiss数据库本地化保存成功！')
    else:
        vector_store = FAISS.load_local(
            "agent_save_local",
            embeddings=embedding_model,
            allow_dangerous_deserialization=True
        )

    #构建检索器
    retriever = vector_store.as_retriever(search_kwargs={"k": 2})

    retriever_tool = create_retriever_tool(
        retriever, "book_retriever",
        description="把时间当朋友"
    )

    #工具列表
    tools = [search, retriever_tool]

    #构建Agent
    agent = create_tool_calling_agent(
        llm=model,
        prompt=prompt,
        tools=tools
    )

    #agent executor
    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True
    )

    while True:
        
        #运行agent
        message = executor.invoke({"input":input("聊点什么呢？")})
        # message = executor.invoke({"input":"总结一下把时间当朋友这本书？"})
        print(message["output"])
        






