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
from langchain_community.document_loaders import PDFMinerLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

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
if not os.path.exists('local_save'):
    # 加载网页中文本内容，转换langchain处理的Document
    loader = PDFMinerLoader("https://storage.googleapis.com/deepmind-media/Era-of-Experience%20/The%20Era%20of%20Experience%20Paper.pdf")
    docs = loader.load()

    # TextSplitter实现加载后Document分割
    splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n','\n',''],
        chunk_size=1000,
        chunk_overlap=100
    )
    splited_docs = splitter.split_documents(docs)
    
    # 创建向量数据库（内存中）对chunk进行向量化和存储
    vector_store = FAISS.from_documents(
        documents=splited_docs,
        embedding=embedding_model)
    # 向量数据库本地化存储
    vector_store.save_local('week16/local_save')
    print('faiss数据库本地化保存成功！')
else:
    vector_store = FAISS.load_local(
        'week16/local_save', 
        embeddings=embedding_model, 
        allow_dangerous_deserialization=True
    )
    print('加载faiss数据库本地化记录成功！')

# 构建检索器
retriever = vector_store.as_retriever(search_kwargs={"k": 2})
vector_store = FAISS.load_local(
        'week16/local_save', 
        embeddings=embedding_model, 
        allow_dangerous_deserialization=True
)
# 构建检索器
retriever_tool = create_retriever_tool(
    retriever, "book_retriever",
    description="围绕《把时间当作朋友》核心内容，从心智力量、时间管理、自律坚持、兴趣与努力的关系、对成功学的反思等多方面，阐述了通过开启和管理心智、合理规划时间、克服懒惰、持续行动等实现个人成长与改变的理念。"
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
msgs = executor.invoke({"input":"请用中文回答：人工智能的未来方向是什么？对于时间管理有什么帮助？"})

print(msgs['output'])