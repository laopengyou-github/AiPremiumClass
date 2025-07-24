#使用langchain_agent实现pdf和TAVILY工具组合动态调用的实现。
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from dotenv import load_dotenv, find_dotenv
from langchain import hub
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.agents import create_tool_calling_agent, AgentExecutor
import os
from langchain_community.vectorstores import FAISS
from langchain.tools.retriever import create_retriever_tool
from langchain_community.document_loaders  import  PDFMinerLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv(find_dotenv())
model = ChatOpenAI(
    model="gpt-3.5-turbo",
    api_key=os.environ['api_key'],
    base_url=os.environ['base_url'])
prompt = hub.pull("hwchase17/openai-functions-agent")
search = TavilySearchResults(max_results=2)
embedding_model = OpenAIEmbeddings(
    model="text-embedding-ada-002",
    api_key=os.environ['api_key'],
    base_url=os.environ['base_url'],
)
if not os.path.exists('local_save'):
    loader = PDFMinerLoader(file_path="./时间的秩序.pdf")
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', ''],
        chunk_size=1000,
        chunk_overlap=100
    )
    splited_docs = splitter.split_documents(docs)
    vector_store = FAISS.from_documents(
        documents=splited_docs,
        embedding=embedding_model
    )
    vector_store.save_local('local_save1')
    print('FAISS数据库本地化保存成功！')
else:
    vector_store = FAISS.load_local(
        'local_save1',
        embeddings=embedding_model,
        allow_dangerous_deserialization=True
    )
    print('加载FAISS数据库本地化记录成功！')

retriever = vector_store.as_retriever(search_kwargs={"k": 2})
retriever_tool = create_retriever_tool(
    retriever, "book_retriever",
    description="从《时间的秩序》这本书中检索相关内容。适用于需要书中具体段落、概念或理论支持的问题，特别是当答案需严格基于书本内容时。"
                "如果问题涉及《时间的秩序》中的具体内容，请使用 book_retriever 工具；如果需要最新或广泛的信息，请使用 tavily_search 工具。")

tools = [search,retriever_tool]
agent = create_tool_calling_agent(
    llm=model, prompt=prompt, tools=tools)
executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True)
msgs = executor.invoke({"input":"《时间的秩序》中提到的分立性是什么？请结合书中内容回答。并通过互联网查找最新的物理学观点支持这一理论"})
print(msgs['output'])





