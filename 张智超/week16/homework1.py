import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_tavily import TavilySearch
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PDFMinerLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools.retriever import create_retriever_tool
from langchain import hub
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate

if __name__ == '__main__':
    load_dotenv()
    # llm模型
    llm_model = ChatOpenAI(
        model="glm-4-flash-250414",
        api_key=os.getenv("MODEL_API_KEY"),
        base_url=os.getenv("MODEL_BASE_URL")
    )
    # 网页搜索工具
    search = TavilySearch(max_results=2)
    # embedding模型
    embed_model = OpenAIEmbeddings(
        model="embedding-3",
        api_key=os.environ['MODEL_API_KEY'],
        base_url=os.environ['MODEL_BASE_URL']
    )
    if not os.path.exists('files/faiss_save'):
        # 加载数据
        loader = PDFMinerLoader(file_path ="./files/论人工智能的可行性.pdf")
        docs = loader.load()
        # 数据切分
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100
        )
        split_docs = text_splitter.split_documents(docs)
        # 创建向量数据库，进行向量化和存储
        vector_store = FAISS.from_documents(
            documents=split_docs,
            embedding=embed_model
        )
        # 向量数据库本地化存储
        vector_store.save_local('files/faiss_save')
        print("faiss数据库本地保存成功")
    else:
        vector_store = FAISS.load_local(
            'files/faiss_save', 
            embeddings=embed_model, 
            allow_dangerous_deserialization=True
        )
        print("faiss数据库本地加载成功")
    # rag检索
    retriever = vector_store.as_retriever(search_kwargs={"k": 2})
    retriever_tool = create_retriever_tool(
        retriever, 
        "ai_knowledge_search",
        description="检索人工智能相关信息，任何人工智能相关的问题，需要使用此工具!"
    )
    # 提示词模版：https://smith.langchain.com/hub/hwchase17/openai-functions-agent
    # prompt = hub.pull("hwchase17/openai-functions-agent")
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a helpful assistant. Answer the question about AI(artificial intelligence) from the provided context, make sure to provide all the details. If the answer is not in
                provided context just say, "answer is not available in the context", don't provide the wrong answer""",
            ),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )
    # 工具列表
    tools = [search, retriever_tool]
    # tools = [retriever_tool]
    # 构建agent
    agent = create_tool_calling_agent(
        llm=llm_model, 
        prompt=prompt, 
        tools=tools
    )
    # agent executor
    executor = AgentExecutor(
        agent=agent, 
        tools=tools, 
        verbose=True # 打印日志
    )
    # 运行agent 
    # msgs = executor.invoke({"input":"青岛明天天气如何"})
    msgs = executor.invoke({"input":"简要说明一下人工智能的应用前景"})

    print(msgs['output'])