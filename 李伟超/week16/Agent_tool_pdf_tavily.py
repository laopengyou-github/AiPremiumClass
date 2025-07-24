from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from dotenv import load_dotenv, find_dotenv
from langchain import hub
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_tavily import TavilySearch

from langchain.agents import create_tool_calling_agent, AgentExecutor
import os
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PDFMinerLoader
from langchain.tools.retriever import create_retriever_tool

load_dotenv()

# 构建LLM
model = ChatOpenAI(
    model="glm-4-flash-250414",
    api_key=os.environ['api_key'],
    base_url=os.environ['base_url']) # 支持function call

# 提示词模版
prompt = hub.pull("hwchase17/openai-functions-agent")

# 构建工具
search = TavilySearch(max_results=2)

embedding_model = OpenAIEmbeddings(
    model="embedding-3",
    api_key=os.environ['api_key'],
    base_url=os.environ['base_url']
)

# 载入 FAISS 向量存储
vector_store = FAISS.load_local(
        'local_save', 
        embeddings=embedding_model, 
        allow_dangerous_deserialization=True
)
# 构建检索器
retriever = vector_store.as_retriever(search_kwargs={"k": 2})
# retriever_tool = create_retriever_tool(
#     retriever, "book_retriever",
#     description="围绕《把时间当作朋友》核心内容，从心智力量、时间管理、自律坚持、兴趣与努力的关系、对成功学的反思等多方面，阐述了通过开启和管理心智、合理规划时间、克服懒惰、持续行动等实现个人成长与改变的理念。"
# )

# 加入PDF加载器并处理PDF文件
pdf_loader = PDFMinerLoader("https://storage.googleapis.com/deepmind-media/Era-of-Experience%20/The%20Era%20of%20Experience%20Paper.pdf")
pdf_documents = pdf_loader.load_and_split()

pdf_retriever_tool = create_retriever_tool(
    retriever, "pdf_retriever", 
    description="The Era of Experience is a paper that discusses the evolution of technology and its impact on human experience, emphasizing the importance of understanding user needs and creating meaningful interactions."
)

# 工具列表
tools = [search,pdf_retriever_tool]

# 构建agent
agent = create_tool_calling_agent(
    llm=model, prompt=prompt, tools=tools)

# agent executor
executor = AgentExecutor(
    agent=agent, 
    tools=tools, 
    verbose=True)

# 运行agent 
msgs = executor.invoke({"input":"What is the Era of Experience?"})

print(msgs['output'])