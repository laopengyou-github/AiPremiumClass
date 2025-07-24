"""
1.使用langchain_agent实现pdf和TAVILY工具组合动态调用的实现。
"""
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from dotenv import load_dotenv, find_dotenv
from langchain import hub
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.agents import create_tool_calling_agent, AgentExecutor
import os
from langchain_community.vectorstores import FAISS
from langchain.tools.retriever import create_retriever_tool

load_dotenv()
# 禁用 LangSmith 追踪以避免线程错误
os.environ["LANGCHAIN_TRACING_V2"] = "false"
# 构建LLM
model = ChatOpenAI(
    model="glm-4-flash-250414",
    api_key=os.environ['ZHIPU_API_KEY'],
    base_url=os.environ['BASE_URL']) # 支持function call

# 提示词模版
prompt = hub.pull("hwchase17/openai-functions-agent")

# 构建工具
search = TavilySearchResults(max_results=2)

embedding_model = OpenAIEmbeddings(
    model="embedding-3",
    api_key=os.environ['ZHIPU_API_KEY'],
    base_url=os.environ['BASE_URL']
)
vector_store = FAISS.load_local(
        './陈兴/week15/local_save', 
        embeddings=embedding_model, 
        allow_dangerous_deserialization=True
)
# 构建检索器
retriever = vector_store.as_retriever(search_kwargs={"k": 2})
retriever_tool = create_retriever_tool(
    retriever, "book_retriever",
    description="The-Era-of-Experience论文"
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
# msgs = executor.invoke({"input":"上海明天天气如何"})
# msgs = executor.invoke({"input":"你可以调用的工具有哪些?"})
msgs = executor.invoke({"input":"什么是 The-Era-of-Experience, 请你查阅本地知识库回答"})

print(msgs['output'])