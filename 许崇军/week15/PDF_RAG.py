from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from dotenv import load_dotenv, find_dotenv
from langchain_community.document_loaders  import  PDFMinerLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain import hub
import os

if __name__ == '__main__':
    load_dotenv(find_dotenv())
    # llm model
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        api_key = os.environ['api_key'],
        base_url=os.environ['base_url'],
        streaming=True
    )
    # embedding model
    embedding_model = OpenAIEmbeddings(
        api_key=os.environ['api_key'],
        base_url=os.environ['base_url']
    )

    if not os.path.exists('pdfRAG_save'):
        loader = PDFMinerLoader(file_path="./The Era of Experience Paper.pdf")
        docs = loader.load()

        # TextSplitter实现加载后Document分割
        splitter = RecursiveCharacterTextSplitter(
            separators=['\n\n', '\n', ''],
            chunk_size=1000,
            chunk_overlap=100
        )
        splited_docs = splitter.split_documents(docs)

        vector_store = FAISS.from_documents(
            documents=splited_docs,
            embedding=embedding_model)
        vector_store.save_local('pdfRAG_save')
        print('faiss数据库本地化保存成功！')
    else:
        vector_store = FAISS.load_local(
            'pdfRAG_save',
            embeddings=embedding_model,
            allow_dangerous_deserialization=True
        )
        print('加载faiss数据库本地化记录成功！')

    # 构建检索器
    retriever = vector_store.as_retriever(search_kwargs={"k": 2})

    def format_docs(docs):
        return '\n\n'.join([doc.page_content for doc in docs])

    prompt = hub.pull('rlm/rag-prompt')
    rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
    )
    response = rag_chain.invoke("how to build a world model?")
    print(response)