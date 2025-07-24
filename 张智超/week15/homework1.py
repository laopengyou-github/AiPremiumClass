import os
from dotenv import load_dotenv, find_dotenv
from langchain_ollama import OllamaEmbeddings
from langchain_community.document_loaders import PDFMinerLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain import hub
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


if __name__ == "__main__":
    load_dotenv(find_dotenv())
    # embedding模型
    embed_model = OllamaEmbeddings(
        model="nomic-embed-text",
        base_url="http://localhost:11434"
    )
    if not os.path.exists('hw1_save'):
        # 加载数据
        loader = PDFMinerLoader(file_path ="./files/TheEraofExperiencePaper.pdf")
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
        vector_store.save_local('hw1_save')
        print("faiss数据库本地保存成功")
    else:
        vector_store = FAISS.load_local(
            'hw1_save', 
            embeddings=embed_model, 
            allow_dangerous_deserialization=True
        )
        print("faiss数据库本地加载成功")

    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    # 将检索的内容拼接起来
    def doc_format(docs):
        doc_text = "\n".join([d.page_content for d in docs])
        print("doc_text=", doc_text)
        return doc_text
    # 大语言模型
    llm = ChatOpenAI(
        model_name="glm-4-flash-250414",
        base_url=os.environ.get('BASE_URL'),
        api_key=os.environ.get('API_KEY'),
    )
    # prompt 
    prompt = hub.pull('rlm/rag-prompt')
    # 构建chain：第一个字典中的key是langchain_hub的占位符
    rag_chain = (
        {"context": retriever | doc_format, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    # 结合rag检索的回答
    response = rag_chain.invoke("请说明一下人类数据时代是什么？")
    print("AI:", response)
