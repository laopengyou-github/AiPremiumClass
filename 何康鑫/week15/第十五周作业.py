# 导入必要库
from langchain_community.document_loaders import PDFMinerLoader
from langchain.text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
import os

# 1. 文档加载
loader = PDFMinerLoader("The_Era_of_Experience_Paper.pdf")  # 替换为实际PDF路径
docs = loader.load()

# 2. 文本分块
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len
)
splits = text_splitter.split_documents(docs)

# 3. 向量存储
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(splits, embeddings)
vectorstore.save_local("pdf_faiss_index")

# 4. 构建检索问答链
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model="gpt-4o-mini"),
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)

# 5. 测试检索
query = "请解释经验时代的主要特征"
answer = qa_chain.invoke({"query": query})
print(answer['answer'])

pip install graphrag langchain langchain-community
import os
from graphrag.index import GraphRagIndexer
from graphrag.config import GraphRagConfig

# 1. 配置参数
config = GraphRagConfig(
    project_name="novel_study",
    data_path="./novel_data",
    storage_path="./graphrag_storage",
    chunk_size=500,
    chunk_overlap=50,
    num_clusters=5,
    max_iterations=3
)

# 2. 准备数据
os.makedirs(config.data_path, exist_ok=True)
with open(os.path.join(config.data_path, "novel.txt"), "w") as f:
    f.write("""
    # 小说内容示例
    第一章 春日午后
    ...
    """)

# 3. 构建索引
indexer = GraphRagIndexer(config)
indexer.run()

# 4. 本地查询
answer = indexer.query(
    method="local",
    query="请分析主角性格发展",
    context_window=2
)
print("本地查询结果:")
print(answer)

# 5. 全局查询
answer = indexer.query(
    method="global",
    query="小说主题是什么",
    summary_depth=3
)
print("全局查询结果:")
print(answer)
