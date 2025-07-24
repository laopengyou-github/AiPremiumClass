
from langchain_community.embeddings import HuggingFaceEmbeddings
import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain.document_loaders import PDFMinerLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.language_models import LLM
from typing import List, Optional
import requests
import os
from langchain_core.runnables import RunnablePassthrough

class ZhipuLLM(LLM):
    model: str = "glm-4-flash"
    api_key: str = os.environ["OPENAI_API_KEY"]
    base_url: str = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
# Example values:"""

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        body = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}]
        }
        response = requests.post(self.base_url, headers=headers, json=body)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

    @property
    def _llm_type(self) -> str:
        return "zhipu"
    
# if __name__ == "__main__":
#     print("🧪 智谱 GLM 测试：")
#     llm = ZhipuLLM()
#     print(llm.invoke("请用一句话总结人工智能的核心目标"))


if __name__ == '__main__':
    # Load environment variables
    load_dotenv()
    # 加载文档

    # 
    llm = ZhipuLLM()

    # llm = ZhipuLLM(
    #     model="glm-4-flash",
    #     api_key=os.environ['OPENAI_API_KEY'],
    #     base_url=os.environ['BASE_URL']
    # )
    
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
        # api_key=os.environ["OPENAI_API_KEY"],
        # base_url=os.environ["BASE_URL"]
    )

    if not os.path.exists('local_save'):
        # Load text document
        loader = PDFMinerLoader("https://storage.googleapis.com/deepmind-media/Era-of-Experience%20/The%20Era%20of%20Experience%20Paper.pdf")
        docs = loader.load()

        # Split documents into chunks
        splitter = RecursiveCharacterTextSplitter(
            separators=['\n\n','\n',''],
            chunk_size=1000,
            chunk_overlap=100
        )
        splited_docs = splitter.split_documents(docs)

        # Create vector store and save locally
        vector_store = FAISS.from_documents(
            documents=splited_docs,
            embedding=embedding_model
        )
        vector_store.save_local('local_save')
        print('FAISS数据库本地化保存成功!')

    else:
        vector_store = FAISS.load_local(
            'local_save', 
            embeddings=embedding_model, 
            allow_dangerous_deserialization=True
        )
        print('加载FAISS数据库本地化记录成功!')

    # Create retriever
    retriever = vector_store.as_retriever(search_kwargs={"k": 2})

    def format_docs(docs):
        return '\n\n'.join([doc.page_content for doc in docs])
    
    prompt = hub.pull("rlm/rag-prompt")

    # Create RetrievalQA chain
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()} 
        | prompt 
        | llm
        | StrOutputParser()
    )

    response = rag_chain.invoke("这篇论文主要讲什么？")
    print(response)
