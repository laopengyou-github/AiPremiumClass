"""
使用langchain_agent实现pdf和TAVILY工具组合动态调用的实现。
"""
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from dotenv import load_dotenv
from langchain import hub
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.agents import create_tool_calling_agent, AgentExecutor
import os
from langchain.document_loaders import PDFMinerLoader
import faiss
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain.tools.retriever import create_retriever_tool
from uuid import uuid4

if __name__ == '__main__':
    
    load_dotenv()

    # 构建LLM
    model = ChatOpenAI(
        model="glm-4-flash-250414",
        api_key=os.environ['OPENAI_API_KEY'],
        base_url=os.environ['OPENAI_API_BASE']) # 支持function call

    # 提示词模版
    prompt = hub.pull("hwchase17/openai-functions-agent")

    # 构建RAG工具
    search = TavilySearchResults(max_results=2)
    embedding_model = OpenAIEmbeddings(
        model="embedding-3",
        api_key=os.environ['OPENAI_API_KEY'],
        base_url=os.environ['OPENAI_API_BASE']
    )

    if os.path.exists("faiss_index"):
        vector_store = FAISS.load_local("faiss_index", embedding_model, allow_dangerous_deserialization=True)
    else:
        loader = PDFMinerLoader("Reasoning to Learn from Latent Thoughts.pdf")
        documents = loader.load()
        # 拆分文档
        docs = RecursiveCharacterTextSplitter(
            chunk_size=1000,    # 每个块的大小
            chunk_overlap=100,  # 块之间的重叠大小
        ).split_documents(documents)
        # 构建向量数据库 
        index = faiss.IndexFlatL2(len(embedding_model.embed_query("hello world")))
        vector_store = FAISS(
            embedding_function=embedding_model,
            index=index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={},
        )
        uuids = [str(uuid4()) for _ in range(len(docs))]

        # 每次添加64个文档，避免超出token限制
        for i in range(0, len(docs), 64):
            batch_docs = docs[i:i+64]
            batch_uuids = uuids[i:i+64]
            vector_store.add_documents(documents=batch_docs, ids=batch_uuids)
        vector_store.save_local("faiss_index")

    # 构建检索器
    retriever = vector_store.as_retriever(search_kwargs={"k": 2})

    retriever_tool = create_retriever_tool(
        retriever, "book_retriever",
        description="文档主要讲了一种能让语言模型更高效学习的新方法，核心思路是让模型 “还原” 文字背后隐藏的思考过程。"
    )

# 工具列表
tools = [search,retriever_tool]

# 构建agent
agent = create_tool_calling_agent(
    llm=model, prompt=prompt, tools=tools)

# agent executor
executor = AgentExecutor(
    agent=agent, 
    tools=tools, 
    verbose=True)

# 运行agent 
# msgs = executor.invoke({"input":"查询一下北京明天天气如何"})
msgs = executor.invoke({"input":"语言模型最新的学习方法是什么？"})

print(msgs['output'])