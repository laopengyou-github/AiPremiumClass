from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from dotenv import load_dotenv, find_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import PDFMinerLoader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from langchain import hub

import os


if __name__ == '__main__':
    load_dotenv()
    # llm model
    llm = ChatOpenAI(
        model="glm-4-flash-250414",
        api_key=os.environ['api_key'],
        base_url=os.environ['base_url']
    )
    # embedding model 
    embedding_model = OpenAIEmbeddings(
        api_key=os.environ['api_key'],
        base_url=os.environ['base_url']
    )

    if not os.path.exists('local_save'):
        # 加载网页中文本内容，转换langchain处理的Document
        #loader = WebBaseLoader(web_path='https://hawstein.com/2014/03/06/make-thiner-make-friend-with-time/')
        #docs = loader.load()
        # 加载PDF中文本内容，转换langchain处理的Document
        loader = PDFMinerLoader(file_path="E:/XUEXIZILIAO/作业/week15/Welcome_to_the_Era_of_Experience.pdf")
        docs = loader.load()
        # TextSplitter实现加载后Document分割
        splitter = RecursiveCharacterTextSplitter(
            separators=['\n\n','\n',''],
            chunk_size=1000,
            chunk_overlap=100
        )
        splited_docs = splitter.split_documents(docs)
        
        # 创建向量数据库（内存中）对chunk进行向量化和存储
        vector_store = FAISS.from_documents(
            documents=splited_docs,
            embedding=embedding_model)
        # 向量数据库本地化存储
        vector_store.save_local('local_save')
        print('faiss数据库本地化保存成功')
    else:
        vector_store = FAISS.load_local(
            'local_save', 
            embeddings=embedding_model, 
            allow_dangerous_deserialization=True
        )
        print('加载faiss数据库本地化记录成功')

    # 构建检索器
    retriever = vector_store.as_retriever(search_kwargs={"k": 2})

    def format_docs(docs):
        return '\n\n'.join([doc.page_content for doc in docs])

    # prompt 
    prompt = hub.pull('rlm/rag-prompt')

    # 构建rag chain
    rag_chain = (
        {"context": retriever | format_docs , "question": RunnablePassthrough()} 
        | prompt 
        | llm 
        | StrOutputParser()
    )

  