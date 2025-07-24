import os
from dotenv import load_dotenv, find_dotenv
from langchain_community.chat_models import ChatTongyi
from langchain_community.document_loaders import PDFMinerLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.agents import create_tool_calling_agent,AgentExecutor
from langchain import hub
from langchain.tools.retriever import create_retriever_tool

def load_pdf_and_index(pdf_path, embedding_model):
    """加载PDF文档并创建FAISS索引"""
    if not os.path.exists("faiss_index_pdf"):
        # 加载PDF文档
        loader = PDFMinerLoader(pdf_path)
        documents = loader.load()
        print(f"成功加载 {len(documents)} 个文档")
        
        # 分割文档
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(documents)
        print(f"分割后文档数量: {len(splits)}")
        
        # 创建FAISS向量存储
        vectorstore = FAISS.from_documents(splits, embedding=embedding_model)
        vectorstore.save_local("faiss_index_pdf")  # 保存到本地目录
        print(f"向量存储已保存到 'faiss_index_pdf' 目录")    
    else:
        # 如果向量存储已存在，直接加载
        vectorstore = FAISS.load_local(
            "faiss_index_pdf", 
            embeddings=embedding_model,  # 使用相同的嵌入模型加载
            allow_dangerous_deserialization=True  # 允许不安全的反序列化
        )
        print(f"从本地加载向量存储，文档数量: {len(vectorstore.docstore._dict)}")
    
    return vectorstore

if __name__ == "__main__":
    # Load environment variables
    load_dotenv(find_dotenv())
    
    # LLM 使用通义千问
    llm = ChatTongyi(
        model="qwen-max",  # 选择模型版本，如 "qwen-long" 或 "qwen-turbo" -- coder模型不行
        top_p=0.8,          # 参数调整，如 top-p 控制生成的多样性
        temperature=0.1,    # 温度参数，控制生成的随机性
        api_key=os.getenv("api_key")
    )

    # embedding模型使用bge-m3
    # embedding_model = HuggingFaceEmbeddings(
    #     model_name="BAAI/bge-m3",  # 模型名称
    #     model_kwargs={'device': 'cpu'},  # 若有GPU，指定'cuda'加速；否则用'cpu'
    #     encode_kwargs={'normalize_embeddings': True}  # 归一化嵌入向量（推荐，提升检索效果）
    # )
    # 使用HuggingFace的嵌入模型，all-MiniLM-L6-v2是一个bi-encoder模型，适合文本嵌入
    embedding_model = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    # Tavily网络搜索工具
    search = TavilySearchResults(max_results=2)

    # RAG文档检索工具-加载PDF文档并创建FAISS索引
    vectorstore = load_pdf_and_index(
        pdf_path="The Era of Experience Paper.pdf", # PDF文档路径
        embedding_model=embedding_model # 使用HuggingFace的嵌入模型
    )
    # Retriever 构建检索器
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2}) # 创建检索器,默认返回2个相关文档
    retriever_tool = create_retriever_tool(
        retriever, "book_retriever",
        description="围绕人工智能体验时代的核心内容，从PDF文档中检索相关信息。"
    )

    # 工具列表
    tools = [
        search,  # 网络搜索工具
        retriever_tool  # 文档检索工具
    ]

    # Agent 构建
    prompt = hub.pull("hwchase17/openai-functions-agent")  # 从Hub中获取RAG提示模板
    # 创建Agent
    agent = create_tool_calling_agent(
        llm=llm,  # 使用通义千问大模型
        prompt=prompt,  # 提示词模板
        tools=tools,  # 工具列表
    )
    # agent executor
    executor = AgentExecutor(
        agent=agent,  # 使用创建的Agent
        tools=tools,  # 工具列表
        verbose=True  # 打印详细信息
    )
    # 执行Agent
    query = {"input":"请简要介绍一下什么是人工智能的体验时代。"}
    print(f"用户查询: {query}")
    response = executor.invoke(query)
    print(f"Agent响应-1: {response['output']}")

    query_2 = {"input":"北京明天天气如何？"}
    print(f"用户查询: {query_2}")
    response_2 = executor.invoke(query_2)
    print(f"Agent响应-2: {response_2['output']}")

    # 用户查询: {'input': '请简要介绍一下什么是人工智能的体验时代。'}
    # > Entering new AgentExecutor chain...
    # Invoking: `book_retriever` with `{'query': '人工智能体验时代'}`
    # > Finished chain.
    # Agent响应-1: 人工智能的体验时代是指人工智能技术发展到一个阶段，其中AI系统能够通过与人类和环境的交互来学习和适应，
    # 从而提供更加个性化、情境化的用户体验。在这个时代，AI不仅 仅是执行特定任务的工具，而是能够理解用户的需求和偏好，
    # 并且随着时间的推移不断优化其性能。...
    
    # 用户查询: {'input': '北京明天天气如何？'}
    # > Entering new AgentExecutor chain...
    # Invoking: `tavily_search_results_json` with `{'query': '北京明天天气'}`
    # > Finished chain.
    # Agent响应-2: 明天北京的天气预报如下：
    # - 白天：多云，南转北风1、2级，最高气温36℃。
    # - 夜间：多云，南转北风1、2级，最低气温21℃。
    # 看起来明天北京不会下雨，且白天温度较高，请注意防晒和补水