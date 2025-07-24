from langchain_openai import ChatOpenAI
from dotenv import load_dotenv, find_dotenv
import os
from langchain_community.document_loaders import PDFMinerLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# 设置环境变量（关键配置）
os.environ.update({
    "USER_AGENT": "AcademicResearchBot/1.0",
    "HF_ENDPOINT": "https://hf-mirror.com",  # 使用镜像服务器
    "HF_HUB_DISABLE_SYMLINKS_WARNING": "1",
    "TF_ENABLE_ONEDNN_OPTS": "0",
    "HF_HOME": "./hf_cache"  # 自定义缓存目录
})

if __name__ == '__main__':
    # 加载环境变量
    load_dotenv(find_dotenv())

    # 初始化语言模型
    model = ChatOpenAI(
        model="glm-4-flash-250414",
        base_url=os.environ['base_url'],
        api_key=os.environ['api_key'],
        temperature=0.7
    )

    # 使用 huggingface 的 bge-small-zh-v1.5 模型进行向量化
    embedding_model = HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-zh-v1.5",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True},
            cache_folder=os.environ["HF_HOME"]
        )

    if not os.path.exists("faiss_index"):
        # 定义pdf路径
        pdf_path = "https://storage.googleapis.com/deepmind-media/Era-of-Experience%20/The%20Era%20of%20Experience%20Paper.pdf"

        # 加载文档，转换 langchain 处理的 document
        loader = PDFMinerLoader(file_path = pdf_path)
        docs = loader.load()

        # 定义文档切分器
        text_splitter = RecursiveCharacterTextSplitter(
            separators = ['\n\n', '\n', ''], # 分隔符优先级从高到低
            chunk_size = 1024,
            chunk_overlap = 128
        )

        # # 切分文档
        splited_docs = text_splitter.split_documents(docs)

        # 创建向量数据库(内存中)，对chunk进行向量化和存储
        # 构建向量数据库
        vector_store = FAISS.from_documents(documents=splited_docs, embedding=embedding_model)

        # 保存向量数据库
        vector_store.save_local(folder_path="faiss_index")
        print("faiss向量数据库保存成功！")
    else:
        # 加载向量数据库
        vector_store = FAISS.load_local(
                folder_path="faiss_index",
                embeddings=embedding_model,
                allow_dangerous_deserialization=True
            )
        print("faiss向量数据库加载成功！")

    # 构建检索器
    retriever = vector_store.as_retriever(search_kwargs={"k": 6})

    docs = retriever.invoke("AI development")

    def format_doc(docs):
        docs_str = '\n\n'.join([doc.page_content for doc in docs])
        return docs_str

    # print(format_doc(docs))

    # 构建 prompt
    prompt = hub.pull("rlm/rag-prompt")

    # 定义输出解析器
    parser = StrOutputParser()

    # 构建 RAG Chain
    rag_chain = (
        {
            'context': retriever | format_doc,
            'question': RunnablePassthrough()
        }
        | prompt
        | model
        | parser
    )

    # rag 检索
    response = rag_chain.invoke("What will the future development of artificial intelligence be like ?")
    print(response)

