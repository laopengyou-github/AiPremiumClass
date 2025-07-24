import os
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_neo4j import Neo4jGraph  # 更新导入语句
from langchain.chains import GraphCypherQAChain

# 加载环境变量
load_dotenv()

# 初始化 OpenAI 模型
llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    api_key=os.getenv("OPENAI_API_KEY")
)

pdf_url = "https://storage.googleapis.com/deepmind-media/Era-of-Experience%20/The%20Era%20of%20Experience%20Paper.pdf"
# 加载文档
loader = PyPDFLoader(pdf_url)
documents = loader.load()

# 文本分块
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(documents)

# 初始化 Neo4j 图数据库
graph = Neo4jGraph(
    url=os.getenv("NEO4J_URI"),
    username=os.getenv("NEO4J_USERNAME"),
    password=os.getenv("NEO4J_PASSWORD")
)

# 构建 GraphRAG 链
chain = GraphCypherQAChain.from_llm(
    llm=llm,
    graph=graph,
    verbose=True
)

# 本地问题示例
local_question = "小王子在玫瑰园里说了什么？"
local_answer = chain.run(local_question)
print(f"本地问题: {local_question}")
print(f"答案: {local_answer}")

# 全局问题示例
global_question = "小王子旅行的主要目的是什么？"
global_answer = chain.run(global_question)
print(f"全局问题: {global_question}")
print(f"答案: {global_answer}")