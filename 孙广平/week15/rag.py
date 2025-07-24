from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PDFMinerLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain import hub
from dotenv import load_dotenv
import os
import requests
from langchain.prompts import PromptTemplate
import re
from langchain_core.embeddings import Embeddings



PDF_URL = "https://storage.googleapis.com/deepmind-media/Era-of-Experience%20/The%20Era%20of%20Experience%20Paper.pdf"
PDF_PATH = "era_of_experience.pdf"
VECTOR_STORE_PATH = "local_pdf_faiss"


class DoubaoEmbedding(Embeddings):
    def __init__(self, base_url, api_key):
        self.base_url = base_url
        self.api_key = api_key

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        headers = {"Authorization": f"Bearer {self.api_key}"}
        payload = {"input": texts, "model": "doubao-embedding-text-240715"}
        resp = requests.post(
            f"{self.base_url}/embeddings",
            json=payload,
            headers=headers
        )
        return [d["embedding"] for d in resp.json()["data"]]

    def embed_query(self, text: str) -> list[float]:
        headers = {"Authorization": f"Bearer {self.api_key}"}
        payload = {"input": [text], "model": "doubao-embedding-text-240715"}
        resp = requests.post(
            f"{self.base_url}/embeddings",
            json=payload,
            headers=headers
        )
        return resp.json()["data"][0]["embedding"]



if __name__ == '__main__':
    load_dotenv()  

    llm = ChatOpenAI(
        model="doubao-seed-1.6-250615",
        api_key=os.environ['api_key'],  
        base_url=os.environ['base_url']  
    )
    embedding_model = DoubaoEmbedding(
        base_url=os.environ["base_url"],
        api_key=os.environ["api_key"]
    )


    if not os.path.exists(PDF_PATH):
        r = requests.get(PDF_URL)
        with open(PDF_PATH, "wb") as f:
            f.write(r.content)
        print("PDF下载完成。")

    # 构建向量库
    if not os.path.exists(VECTOR_STORE_PATH):
        loader = PDFMinerLoader(PDF_PATH)
        documents = loader.load()
        
        # 文本分割
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(documents)
        
        # 创建并保存向量库
        vector_store = FAISS.from_documents(splits, embedding_model)
        vector_store.save_local(VECTOR_STORE_PATH)
        print("向量库创建完成。")
    else:
        # 加载现有向量库
        vector_store = FAISS.load_local(VECTOR_STORE_PATH, embedding_model, allow_dangerous_deserialization=True)
        print("向量库加载完成。")

    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    def format_docs(docs):
        return '\n\n'.join([doc.page_content for doc in docs])

    prompt = PromptTemplate.from_template(
    """你是一个专业的研究助手，请根据提供的文献内容回答用户的问题。请确保内容严谨且来源于上下文中。
    上下文：
    {context}
    问题：
    {question}
    请基于以上信息用中文作答，（若无法从上下文获取答案，请如实说明）。
    """
    )

    rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
    )

    while True:
        question = input("<question>：")
        if question.strip().lower() in ['q']:
            break
        response = rag_chain.invoke(question)
        print(f"<answer>：{response}")
        print('******************')