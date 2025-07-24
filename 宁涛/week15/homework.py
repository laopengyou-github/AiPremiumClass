import os
from dotenv import find_dotenv, load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from langchain_community.document_loaders import WebBaseLoader, PDFMinerLoader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from langchain import hub
from langchain_core.runnables import RunnablePassthrough

from langchain_core.output_parsers import StrOutputParser


if __name__ == '__main__':
    load_dotenv(find_dotenv())

    model = ChatOpenAI(
        model='glm-4-flash-250414',
        api_key=os.environ['OPENAI_API_KEY'],
        base_url=os.environ['BASE_URL'],
        temperature=0,
    )

    embedding_model = OpenAIEmbeddings(
        model='embedding-3',
        api_key=os.environ['OPENAI_API_KEY'],
        base_url=os.environ['BASE_URL']
    )

    if not os.path.exists('local_save'):
        loader = PDFMinerLoader(web_path='https://storage.googleapis.com/deepmind-media/Era-of-Experience /The Era of Experience Paper.pdf')
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            separators=['\n\n','\n',''],
            chunk_size=1000,
            chunk_overlap=100
        )
        splited_docs = splitter.split_documents(docs)
        
        vector_store = FAISS.from_documents(
            documents=splited_docs,
            embedding=embedding_model
        )

        vector_store.save_local('local_save')
        print('faiss数据库本地化保存成功！')
    else:
        vector_store = FAISS.load_local(
            'local_save',
            embeddings=embedding_model,
            allow_dangerous_deserialization=True
        )
        print('加载faiss数据库本地化记录成功！')
    
    retriever = vector_store.as_retriever(search_kwargs={'k':4})

    def format_docs(docs):
        return '\n\n'.join([doc.page_content for doc in docs])

    
    prompt = hub.pull('rlm/rag-prompt')
    rag_chain = (
        {'context':retriever | format_docs, 'question':RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )

    response = rag_chain.invoke("What if experiential agents could lean from extemal events and signals, and not just human preferences?")
    print(response)
