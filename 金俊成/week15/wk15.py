import os

from dotenv import load_dotenv
from langchain import hub
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai.chat_models.base import BaseChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

if __name__ == '__main__':
    load_dotenv()
    # llm model
    llm = BaseChatOpenAI(
        model='Qwen/Qwen2.5-7B-Instruct',
        openai_api_key=os.environ['OPENAI_API_KEY'],
        openai_api_base=os.environ['BASE_URL'],
        max_tokens=1024
    )
    # embedding model
    embeddings = DashScopeEmbeddings(
        model="text-embedding-v1",
        dashscope_api_key=os.environ['DASHSCOPE_API_KEY']
    )
    # 定义分词器
    # TextSplitter实现加载后Document分割
    splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', ''],
        chunk_size=100,
        chunk_overlap=100
    )
    docs = PDFPlumberLoader('test.pdf').load()
    splitter_docs = splitter.split_documents(docs)
    # 创建向量数据库
    vector_store = FAISS.from_documents(docs, embeddings)
    # 构建检索器
    retriever = vector_store.as_retriever(search_kwargs={'k': 2})


    def format_docs(docs):
        return '\n\n'.join([doc.page_content for doc in docs])


    # prompt
    prompt = hub.pull('rlm/rag-prompt')

    # 构建rag chain
    rag_chain = (
            {'context': retriever | format_docs, 'question': RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
    )

    # response = rag_chain.invoke('当前AI发展的局限性')
    response = rag_chain.invoke('经验时代的机遇与挑战')
    print(response)
