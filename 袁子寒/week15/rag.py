from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from dotenv import load_dotenv, find_dotenv
from langchain_community.document_loaders import PDFMinerLoader
from langchain_community.embeddings import ZhipuAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from openai import OpenAI
from langchain import hub
import os
from zhipuai import ZhipuAI
if __name__ == '__main__':
    load_dotenv()
    # llm model
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        api_key=os.environ['API_KEY'],
        base_url=os.environ['BASE_URL']
    )
    # embedding model 
    embedding = OpenAIEmbeddings(
        api_key=os.environ['API_KEY'],
        base_url=os.environ['BASE_URL']
    )
    # PDF文件路径
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
            embedding=embedding
        )
        vector_store = embedding.embed_documents(splited_docs)
        # 向量数据库本地化存储
        vector_store.save_local('local_save')
        print('faiss数据库本地化保存成功！')
    else:
        vector_store = FAISS.load_local(
            'local_save', 
            embeddings=embedding, 
            allow_dangerous_deserialization=True
        )
        print('加载faiss数据库本地化记录成功！')

    # 构建检索器
    retriever = vector_store.as_retriever(search_kwargs={"k": 7})

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

    # rag检索 - 可以根据PDF内容调整问题
    response = rag_chain.invoke("什么是体验时代？")
    print(response)