import os

from langchain import hub
from langchain.vectorstores import Chroma
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.tools.retriever import create_retriever_tool
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.document_loaders import WebBaseLoader, PDFMinerLoader


model = ChatOpenAI(
    model="gpt-4o",
    api_key=os.environ['api_key'],
    base_url=os.environ['base_url'])

embedding_model = OpenAIEmbeddings(
    api_key=os.environ['api_key'],
    base_url=os.environ['base_url']
)

prompt = hub.pull("hwchase17/openai-functions-agent")

# Tavily
search = TavilySearchResults(
    tavily_api_key=os.environ['tavily_api_key'],
    max_results=2
)

# FAISS
loader = WebBaseLoader(
    web_paths=("https://hawstein.com/2014/03/06/make-thiner-make-friend-with-time/",),
)

docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)
vector_store = FAISS.from_documents(splits, embedding=embedding_model)
retriever = vector_store.as_retriever(search_kwargs={"k": 2})
retriever_tool_rag = create_retriever_tool(
    retriever, "book_retriever",
    description="围绕《把时间当作朋友》核心内容，从心智力量、时间管理、自律坚持、兴趣与努力的关系、对成功学的反思等多方面，阐述了通过开启和管理心智、合理规划时间、克服懒惰、持续行动等实现个人成长与改变的理念。"
)

# pdf
docs = PDFMinerLoader("./邪王真眼/week15/The Era of Experience Paper.pdf").load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

vectorstore = Chroma.from_documents(documents=splits, embedding=embedding_model)
retriever = vectorstore.as_retriever()
retriever_tool_pdf = create_retriever_tool(
    retriever, "pdf_retriever",
    description="We stand on the threshold of a new era in artificial intelligence that promises to achieve an unprecedented level of ability. A new generation of agents will acquire superhuman capabilities by learning predominantly from experience. This note explores the key characteristics that will define this upcoming era."
)

# run
tools = [search, retriever_tool_rag, retriever_tool_pdf]

agent = create_tool_calling_agent(
    llm=model, prompt=prompt, tools=tools)

executor = AgentExecutor(
    agent=agent, 
    tools=tools, 
    verbose=True)

msgs = executor.invoke({"input":"北京明天天气如何"})
print(msgs['output'])
msgs = executor.invoke({"input":"围绕《把时间当作朋友》，说明如何做好时间管理"})
print(msgs['output'])
msgs = executor.invoke({"input":"“体验时代”的特征有哪些？"})
print(msgs['output'])