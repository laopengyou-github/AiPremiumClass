from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from dotenv import load_dotenv
from langchain_community.document_loaders import PDFMinerLoader
from langchain_community.vectorstores import FAISS

import os
from langchain import hub

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

if __name__ == "__main__":
    load_dotenv()
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        api_key=os.environ("OPENAI_API_KEY"),
        base_url=os.environ["BASE_URL"]
    )

    embedding_model = OpenAIEmbeddings(
        api_keys=os.environ["OPENAI_API_KEY"],
        base_url=os.environ["BASE_URL"]
    )

if not os.path.exists("local_save"):
    loader = PDFMinerLoader("The Era of Experience Paper.pdf")
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        SEPARATORS=["\n\n","\n",""],
        chunk_size = 1000,
        chunk_overlap = 100
    )
    splited_docs = splitter.split_documents(docs)

    vector_store = FAISS.from_documents(
        documents=splited_docs,
        embedding=embedding_model
    )

    vector_store = FAISS.load_local("local_save")
    print("faiss本地保存成功")
else:
    vector_store = FAISS.load_local(
        "local_save",
        embedding_model=embedding_model,
        allow_dangerous_deserialization=True
    )
    print("加载faiss数据")


    retriever = vector_store.as_retriever(search_kwargs={"k":2})

    def format_docs(docs):
        return "\n\n".join([doc.page_content for doc in docs])

    prompt = hub.pull("rlm/rag-prompt")

    rag_chain = (
        {"content":retriever| format_docs,"quenstion":RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    response = rag_chain.invoike("请用中文回答一下什么是人工智能体验时代")
