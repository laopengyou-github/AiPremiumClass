"""
1. LLM model
"""
#安装包 pip install langchain langchainhub langchain_community
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from dotenv import load_dotenv, find_dotenv
from langchain import hub
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.agents import create_tool_calling_agent, AgentExecutor
import os
from langchain_community.vectorstores import FAISS
from langchain.tools.retriever import create_retriever_tool

load_dotenv()


os.environ["TAVILY_API_KEY"] = "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"

# 构建LLM
model = ChatOpenAI(
    model="glm-4-flash-250414",
    api_key=os.environ['api_key'],
    base_url=os.environ['base_url']) # 支持function call

# 提示词模版


prompt = hub.pull("hwchase17/openai-functions-agent")
# 构建工具
search = TavilySearchResults(max_results=2)

embedding_model = OpenAIEmbeddings(
    model="embedding-3",
    api_key=os.environ['api_key'],
    base_url=os.environ['base_url']
)
vector_store = FAISS.load_local(
        'local_save', 
        embeddings=embedding_model, 
        allow_dangerous_deserialization=True
)
# 构建检索器
retriever = vector_store.as_retriever(search_kwargs={"k": 2})
retriever_tool = create_retriever_tool(
    retriever, "book_retriever",
    description="PDF文件《RAG相关技术及Agent应用》的相关内容"
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
# msgs = executor.invoke({"input":"北京明天天气如何"})
msgs = executor.invoke({"input":"RAG相关技术及Agent应用"})

print(msgs['output'])