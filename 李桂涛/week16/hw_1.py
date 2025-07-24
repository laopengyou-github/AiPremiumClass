from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from dotenv import load_dotenv, find_dotenv
from langchain import hub
from langchain_community.tools.tavily_search import TavilySearchResults

import os
from langchain_community.vectorstores import FAISS
from langchain.tools.retriever import create_retriever_tool

#加载环境
load_dotenv(find_dotenv())

#构建LLM
model = ChatOpenAI(
    model="glm-4-flash-250414",
    api_key=os.environ['api_key'],
    base_url=os.environ['base_url']
)
#提示词模板 （通用提示词）
from langchain import hub
prompt = hub.pull("hwchase17/react")

# 获取其他必要的 API 密钥
# tavily_api_key = os.getenv("TAVILY_API_KEY")

#构建工具
# from langchain_community.tools.tavily_search import TavilySearchResults
# search =TavilySearchResults(max_results=5)
from langchain_tavily import TavilySearch
search = TavilySearch(max_results=5)
# search = TavilySearch(api_key=tavily_api_key, max_results=5)
# print(search.invoke("北京天气如何"))

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
    description="围绕《把时间当作朋友》核心内容，从心智力量、时间管理、自律坚持、兴趣与努力的关系、对成功学的反思等多方面，阐述了通过开启和管理心智、合理规划时间、克服懒惰、持续行动等实现个人成长与改变的理念。"
)

#创建工具 列表

tools = [search,retriever_tool]

#构建agents
from langchain.agents import create_tool_calling_agent,AgentExecutor
agent = create_tool_calling_agent(llm=model, prompt=prompt, tools=tools)

# executor agent运行,实际调用获取executor对象,verbose可视化中间日志
executor = AgentExecutor(agent=agent,tools=tools,verbose=True)

# 运行agent
msgs = executor.invoke({"input": "上海天气如何"})
# msgs = executor.invoke({"input":"围绕《把时间当作朋友》，说明如何做好时间管理"})

print(msgs)
