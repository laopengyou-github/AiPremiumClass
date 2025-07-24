"""
1. LLM model
"""
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from dotenv import load_dotenv, find_dotenv
from langchain import hub
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.agents import create_tool_calling_agent, AgentExecutor
import os
from langchain_community.vectorstores import FAISS
from langchain.tools.retriever import create_retriever_tool
import numpy
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
print(numpy.__file__)
from langchain_community.document_loaders import PDFMinerLoader

load_dotenv()

# 构建LLM
model = ChatOpenAI(
    model="glm-4-flash-250414",
    api_key=os.environ['api_key'],
    base_url=os.environ['base_url']) # 支持function call

# 提示词模版
prompt = hub.pull("hwchase17/openai-functions-agent")
# 构建工具
search = TavilySearchResults(max_results=2)

embedding_model = OpenAIEmbeddings( 
    model="embedding-3",
    api_key=os.environ['api_key'],
    base_url=os.environ['base_url']
)
pdf_path = "homework//week15//The Era of Experience Paper.pdf"  # 使用当前目录下的PDF文件
    
if not os.path.exists('local_save'):
    # 使用PDFMinerLoader加载PDF文件内容，转换langchain处理的Document
    loader = PDFMinerLoader(pdf_path)
    docs = loader.load()

    # TextSplitter实现加载后Document分割
    splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n','\n',''],
        chunk_size=1000,
        chunk_overlap=100
    )
    splited_docs = splitter.split_documents(docs)
    
    # 构建向量数据库
    vector_store = FAISS.from_documents(
        documents=splited_docs,  # 这里是 langchain 的 Document 对象列表
        embedding=embedding_model
    )
    # 向量数据库本地化存储
    vector_store.save_local('local_save')
    print('faiss数据库本地化保存成功！')
else:
    vector_store = FAISS.load_local(
        'local_save', 
        embeddings=embedding_model, 
        allow_dangerous_deserialization=True
    )
    print('加载faiss数据库本地化记录成功！')


vector_store = FAISS.load_local(
        'local_save', 
        embeddings=embedding_model, 
        allow_dangerous_deserialization=True
)
# 构建检索器
retriever = vector_store.as_retriever(search_kwargs={"k": 2})
retriever_tool = create_retriever_tool(
    retriever, "book_retriever",
    description="探讨了人工智能即将进入的“体验时代”，即AI将通过与环境互动自主获取经验来超越人类数据限制，实现超人类能力。"
)

# 工具列表
# tools = [search]
tools = [search,retriever_tool]

# 构建agent
agent = create_tool_calling_agent(
    llm=model, prompt=prompt, tools=tools)

# agent executor
executor = AgentExecutor(
    agent=agent, 
    tools=tools, 
    verbose=True)

# 运行agent 
# msgs = executor.invoke({"input":"北京明天天气如何"})
msgs = executor.invoke({"input":"人工智能即将进入的“体验时代”是什么？"})

print(msgs['output'])