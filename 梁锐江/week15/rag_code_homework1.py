import os

from dotenv import load_dotenv
from langchain.embeddings.base import Embeddings
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatZhipuAI
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from zhipuai import ZhipuAI
from langchain_community.document_loaders import PDFMinerLoader


# 自定义 Embedding 类
class ZhipuEmbeddings(Embeddings):
    def __init__(self, model="embedding-2", api_key=None):
        self.client = ZhipuAI(api_key=api_key or os.getenv("API_KEY"))
        self.model = model

    def embed_documents(self, texts):
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text):
        response = self.client.embeddings.create(
            model=self.model,
            input=text
        )
        return response.data[0].embedding


if __name__ == '__main__':
    load_dotenv()

    # llm model
    llm = ChatZhipuAI(
        model_name="glm-4-flash-250414",
        api_key=os.environ['API_KEY'],
        base_url=os.environ['BASE_URL']
    )
    # embedding model
    embedding_model = ZhipuEmbeddings(
        model="embedding-2",
        api_key=os.environ["API_KEY"]
    )

    if not os.path.exists('local_save'):
        # loader = WebBaseLoader(web_path="https://hawstein.com/2014/03/06/make-thiner-make-friend-with-time/")
        loader = PDFMinerLoader(file_path="The Era of Experience Paper.pdf")
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(
            separators=['\n\n', '\n', ''],  # 块分隔符
            chunk_size=1000,  # 分块大小
            chunk_overlap=100  # 分块重叠
        )
        splited_docs = splitter.split_documents(docs)

        # 创建向量数据库（内存中）对chunk进行向量化和存储
        vector_store = FAISS.from_documents(
            documents=splited_docs,
            embedding=embedding_model)

        vector_store.save_local('local_save')
        print('faiss数据库本地化保存成功！')
    else:
        vector_store = FAISS.load_local(
            'local_save',
            embeddings=embedding_model,
            allow_dangerous_deserialization=True
        )
        print('加载faiss数据库本地化记录成功！')

    retriever = vector_store.as_retriever()


    def format_docs(docs):
        return '\n\n'.join([doc.page_content for doc in docs])


    prompt = PromptTemplate.from_template("你是一个助手。\n"
                                          "根据以下内容回答问题：\n{context}\n\n"
                                          "问题：{question}"
                                          )

    rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
    )

    # rag检索
    response = rag_chain.invoke("文件讲述了什么内容")
    print(response)
