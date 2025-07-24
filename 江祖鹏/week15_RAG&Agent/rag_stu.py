from dotenv import load_dotenv, find_dotenv
import os

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import WebBaseLoader,PDFMinerLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

if __name__ == "__main__":

    os.environ["USER_AGENT"] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36 NetType/WIFI MicroMessenger/7.0.20.1781(0x6700143B) WindowsWechat(0x63090c33) XWEB/13907 Flue"

    #加载环境变量
    load_dotenv(find_dotenv())

    llm = ChatOpenAI(
        model = "glm-4-flash-250414",
        base_url = os.environ["base_url"],
        api_key = os.environ["api_key"],
        temperature = 0.1,
    )

    embedding_model = OpenAIEmbeddings(
        model = "Embedding-3",
        base_url = os.environ["base_url"],
        api_key = os.environ["embedding_api_key"],
    )

    #检查本地是否存在向量数据库
    if not os.path.exists("local_save"):
        print("本地向量数据库不存在，开始创建...")
        #加载网页
        loader = PDFMinerLoader("https://storage.googleapis.com/deepmind-media/Era-of-Experience%20/The%20Era%20of%20Experience%20Paper.pdf")
        docs = loader.load()

        #textsplitter加载document后分割
        splitter = RecursiveCharacterTextSplitter(
            separators=['\n\n', '\n', ''],
            chunk_size=1000,
            chunk_overlap=100
        )

        splited_docs = splitter.split_documents(docs)

        #创建向量数据库（内存中）对chunk进行向量化和存储
        vector_store = FAISS.from_documents(
            documents=splited_docs,
            embedding=embedding_model
        )

        #向量数据库本地化存储
        vector_store.save_local("local_save")
        print("faiss数据库本地化保存成功！")
    else:
        print("加载本地向量数据库...")
        vector_store = FAISS.load_local(
            "local_save",
            embeddings=embedding_model,
            allow_dangerous_deserialization=True
        )
        print("加载faiss数据库本地化记录成功！")

    #构建检索器
    retriever = vector_store.as_retriever(search_kwargs={"k": 2})

    # 定义文档格式化函数
    def format_docs(docs):
        return '\n\n'.join([doc.page_content for doc in docs])

    # prompt
    prompt = hub.pull("rlm/rag-prompt")

    # 构建RAG链
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()} 
        | prompt 
        | llm 
        | StrOutputParser()
    )

    # RAG检索
    response = rag_chain.invoke("概括一下这篇论文的主要内容")
    print(response)
