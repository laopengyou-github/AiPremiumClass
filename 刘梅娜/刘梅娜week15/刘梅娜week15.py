from dotenv import find_dotenv, load_dotenv
import os
import requests
import tempfile
import time
from openai import APIStatusError
import asyncio

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    SystemMessage,
    trim_messages
)
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from operator import itemgetter

from langchain_community.document_loaders import PDFMinerLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS


# 封装重试逻辑
def retry_with_backoff(func, max_retries=3, backoff_time=15):
    retries = 0
    while retries < max_retries:
        try:
            return func()
        except APIStatusError as e:
            if e.status_code == 439:
                print(f"请求频率过高，等待 {backoff_time} 秒后重试 ({retries + 1}/{max_retries})")
                time.sleep(backoff_time)
                retries += 1
            else:
                raise
        except Exception as e:
            print(f"执行操作时发生错误: {e}，重试 ({retries + 1}/{max_retries})")
            time.sleep(backoff_time)
            retries += 1
    raise Exception("达到最大重试次数，操作失败")


# 异步创建向量存储
async def create_vectorstore_async(splits, embeddings):
    max_retries = 3
    retries = 0
    while retries < max_retries:
        try:
            vectorstore = await FAISS.from_documents(splits, embeddings)
            return vectorstore
        except APIStatusError as e:
            if e.status_code == 439:
                print(f"请求频率过高，等待 15 秒后重试 ({retries + 1}/{max_retries})")
                await asyncio.sleep(15)
                retries += 1
            else:
                raise
        except Exception as e:
            print(f"创建向量存储时发生错误: {e}，重试 ({retries + 1}/{max_retries})")
            await asyncio.sleep(15)
            retries += 1
    raise Exception("达到最大重试次数，创建向量存储失败")


# 下载 PDF 文件
def download_pdf(pdf_url):
    def _download():
        response = requests.get(pdf_url, timeout=30)
        response.raise_for_status()
        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        try:
            tmp_file.write(response.content)
            tmp_file.flush()
            return tmp_file.name
        finally:
            tmp_file.close()
    return retry_with_backoff(_download)


# 加载 PDF 文件
def load_pdf(path):
    def _load():
        return PDFMinerLoader(path).load()
    return retry_with_backoff(_load)


if __name__ == '__main__':
    load_dotenv(find_dotenv())

    # Initialize the OpenAI Chat model
    model = ChatOpenAI(
        model="gpt-4o-mini",
        base_url=os.environ['BASE_URL'],
        api_key=os.environ['OPENAI_API_KEY'],
        temperature=0.7
    )

    trimmer = trim_messages(
        max_tokens=65,
        strategy="last",
        token_counter=model,
        include_system=True,
        allow_partial=False,
        start_on="human",
    )

    messages = [
        SystemMessage(content="你是一个优秀的助手"),
        HumanMessage(content="你好！我是Bob"),
        AIMessage(content="你好!"),
        HumanMessage(content="我喜欢香草冰激凌"),
        AIMessage(content="赞！"),
        HumanMessage(content="2 + 2等于多少"),
        AIMessage(content="4"),
        HumanMessage(content="谢谢"),
        AIMessage(content="不客气!"),
        HumanMessage(content="要开心呦?"),
        AIMessage(content="是的!"),
    ]

    pdf_url = "https://storage.googleapis.com/deepmind-media/Era-of-Experience%20/The%20Era%20of%20Experience%20Paper.pdf"
    path = download_pdf(pdf_url)

    try:
        docs = load_pdf(path)
    finally:
        # 确保临时文件被删除
        if os.path.exists(path):
            os.remove(path)

    # 文本分块
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    # 创建向量存储所需的嵌入
    embeddings = OpenAIEmbeddings(
        base_url=os.environ['BASE_URL'],
        api_key=os.environ['OPENAI_API_KEY']
    )

    # 异步创建向量存储
    vectorstore = asyncio.run(create_vectorstore_async(splits, embeddings))

    # 定义检索器
    retriever = vectorstore.as_retriever()

    # 构建 RAG 链
    prompt = ChatPromptTemplate.from_messages([
        ("system", "使用以下上下文信息回答用户问题。上下文：{context}"),
        ("human", "{question}")
    ])

    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )

    # 提问示例
    question = "文档中主要讨论了什么内容？"
    def run_rag_chain():
        return rag_chain.invoke(question)

    answer = retry_with_backoff(run_rag_chain)
    print(f"问题: {question}")
    print(f"答案: {answer}")

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "你是一个得力的助手。尽你所能使用{lang}回答所有问题。"),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )

    chain = (
        RunnablePassthrough.assign(messages=itemgetter("messages") | trimmer)
        | prompt
        | model
    )

    def run_chain():
        return chain.invoke(
            {
                "messages": messages + [HumanMessage(content="我叫什么名字")],
                "lang": 'English',
            }
        )

    reponse = retry_with_backoff(run_chain)
    print(reponse.content)