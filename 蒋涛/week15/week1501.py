"""
1. 根据课堂RAG示例，完成外部文档导入并进行RAG检索的过程。
外部PDF文档：https://storage.googleapis.com/deepmind-media/Era-of-Experience%20/The%20Era%20of%20Experience%20Paper.pdf

# 使用 langchain_community.document_loaders.PDFMinerLoader 加载 PDF 文件。
docs = PDFMinerLoader(path).load()

"""

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from dotenv import load_dotenv, find_dotenv
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
        api_key=os.environ['API_KEY'],
        base_url=os.environ['BASE_URL']
    )
    # embedding model 
    embedding_model = OpenAIEmbeddings(
        model="Embedding-3",
        api_key=os.environ['API_KEY'],
        base_url=os.environ['BASE_URL']
    )

    if not os.path.exists('local_save'):
        # 加载网页中文本内容，转换langchain处理的Document
        loader = PDFMinerLoader("https://storage.googleapis.com/deepmind-media/Era-of-Experience%20/The%20Era%20of%20Experience%20Paper.pdf")
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
        vector_store.save_local('week15/local_save')
        print('faiss数据库本地化保存成功！')
    else:
        vector_store = FAISS.load_local(
            'week15/local_save', 
            embeddings=embedding_model, 
            allow_dangerous_deserialization=True
        )
        print('加载faiss数据库本地化记录成功！')

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

    # rag检索
    response = rag_chain.invoke("请用中文回答：人工智能的体验时代代表着人工智能进入是什么阶段")
    print(response)