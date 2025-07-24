from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from dotenv import load_dotenv
from langchain_community.document_loaders import PDFMinerLoader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from langchain_core.prompts import ChatPromptTemplate

import os

# 禁用 LangSmith 追踪以避免线程错误
os.environ["LANGCHAIN_TRACING_V2"] = "false"

if __name__ == '__main__':
    load_dotenv()
    # llm model
    llm = ChatOpenAI(
        model="glm-4-flash-250414",
        api_key=os.environ['ZHIPU_API_KEY'],
        base_url=os.environ['BASE_URL']
    )
    # embedding model 
    embedding_model = OpenAIEmbeddings(
        model="embedding-3",
        api_key=os.environ['ZHIPU_API_KEY'],
        base_url=os.environ['BASE_URL']
    )

    if not os.path.exists('./陈兴/week15/local_save'):
        # 加载PDF文件内容，转换为langchain处理的Document
        loader = PDFMinerLoader('./陈兴/week15/data/The-Era-of-Experience-Paper.pdf')
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
        vector_store.save_local('./陈兴/week15/local_save')
        print('faiss数据库本地化保存成功！')
    else:
        vector_store = FAISS.load_local(
            './陈兴/week15/local_save', 
            embeddings=embedding_model, 
            allow_dangerous_deserialization=True
        )
        print('加载faiss数据库本地化记录成功！')

    # 构建检索器
    retriever = vector_store.as_retriever(search_kwargs={"k": 2})

    def format_docs(docs):
        return '\n\n'.join([doc.page_content for doc in docs])

    # prompt 
    prompt = ChatPromptTemplate.from_template("""You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.

Question: {question} 

Context: {context} 

Answer:""")

    # 构建rag chain
    rag_chain = (
        {"context": retriever | format_docs , "question": RunnablePassthrough()} 
        | prompt 
        | llm 
        | StrOutputParser()
    )

    # rag检索
    response = rag_chain.invoke("What is the main idea of the paper? please respond in English and in Chinese")
    print(response)