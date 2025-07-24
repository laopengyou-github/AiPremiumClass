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
from langchain_community.document_loaders import PDFMinerLoader
from langchain_community.document_loaders import PDFMinerLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain import hub
import os
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
#结合之前创造本地PDF的手段
if not os.path.exists('local_save'):
    # 加载网页中文本内容，转换langchain处理的Document
    docs = PDFMinerLoader('E:\AI\pytorch_class\week15').load()
    # TextSplitter实现加载后Document分割
    splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n','\n',''],
        chunk_size=1000,
        chunk_overlap=100
    )
    splited_docs = splitter.split_documents(docs)
    vector_store = FAISS.from_documents(
            documents=splited_docs,
            embedding=embedding_model)
        # 向量数据库本地化存储
    vector_store.save_local('local_save')
    print('faiss数据库本地化保存成功！')

vector_store = FAISS.load_local(
        'local_save',
        embeddings=embedding_model,
        allow_dangerous_deserialization=True
)
# 构建检索器
retriever = vector_store.as_retriever(search_kwargs={"k": 2})
retriever_tool = create_retriever_tool(
    retriever, "book_retriever",
    description="问题主要负责介绍西游故事"
)

# 工具列表
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
msgs = executor.invoke({"input":"围绕九九八十一难，讲一下你印象里最深的几难"})
print(msgs['output'])