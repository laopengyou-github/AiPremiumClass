"""
    基础RAG分为：构建索引1~3、检索和生成4~5
1、Load：首先使用document_loaders来加载数据(还有web连接或者pdf的加载函数)
2、split：文本数据太大了，先拆分为小块(chunks)
3、store：需要用某种方式来存储和索引我们拆分出来的小块chunks，方便我们检索，一般用vectorstore，embedding模型
4、retrieve：根据用户输入，使用retrieve来检索相关拆分的东西 as_retrieve
5、genetate：使用流 的方式进行对问题进行传递，生成答案
"""
import os
from langchain_openai import ChatOpenAI
from dotenv import find_dotenv,load_dotenv
from langchain_community.vectorstores import FAISS
from langchain import hub
from langchain.document_loaders import WebBaseLoader,PDFMinerLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import langchain_community

load_dotenv(find_dotenv())

# ll = langchain_community.chat_models.ChatZhipuAI.embedding

llm = ChatOpenAI(
    model="glm-4-flash-250414",
    base_url=os.environ['base_url'],
    api_key=os.environ['api_key'],
    temperature=0.75
)

# 加载文档
# loader = WebBaseLoader(web_path="https://storage.googleapis.com/deepmind-media/Era-of-Experience%20/The%20Era%20of%20Experience%20Paper.pdf")
loader = PDFMinerLoader(file_path="./The Era of Experience Paper.pdf")

docs = loader.load()
#定义分割chunk的方法   相邻块之间有 200 个字符的重叠。
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
#进行分割
splits = text_splitter.split_documents(docs)
#将信息进行存储
vectorstore = FAISS.from_documents(documents=splits,embedding= OpenAIEmbeddings(api_key=os.environ['api_key']))

#进行检索和生成
retriever = vectorstore.as_retriever(serch_type="similarity",search_kwargs={"k":3})
prompt = hub.pull("rlm/rag-prompt")

#  用于将检索到的文档格式化为一个字符串
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context":retriever | format_docs,"question":RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

#调用 invoke

rag_chain.invoke("What are some things that can be experienced?")