from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from dotenv import load_dotenv, find_dotenv
from langchain_community.document_loaders import PDFMinerLoader,WebBaseLoader

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
        model="embedding-3",  # 指定模型名称
        api_key=os.environ['api_key'],
        base_url=os.environ['base_url'],
        dimensions=1024  # 按需调整向量维度（若模型支持）
    )
    # 检查嵌入向量维度
    sample_embedding = embedding_model.embed_query("test")
    print(f"向量维度: {len(sample_embedding)}")

    # embedding_model = OpenAIEmbeddings(
    #     model='embedding-2',
    #     dimensions=1024,
    #     api_key=os.environ['api_key'],
    #     base_url=os.environ['base_url']
    # )

    if not os.path.exists('local_save'):
        # 加载网页中文本内容，转换langchain处理的Document
        loader = PDFMinerLoader(file_path='https://storage.googleapis.com/deepmind-media/Era-of-Experience%20/The%20Era%20of%20Experience%20Paper.pdf')
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
        print('faiss数据库本地化保存成功！')
    else:
        vector_store = FAISS.load_local(
            'local_save', 
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
    response = rag_chain.invoke("what is agents?why is it important?")
    print(response)
