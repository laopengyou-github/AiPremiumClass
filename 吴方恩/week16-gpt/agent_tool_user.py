"""
1. LLM model
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
file_path = os.path.join(os.path.dirname(__file__), 'local_save')
pdf_path = os.path.join(os.path.dirname(__file__), 'The Era of Experience Paper.pdf')

if not os.path.exists(file_path):
    # 加载网页中文本内容，转换langchain处理的Document
    loader = PDFMinerLoader(file_path=pdf_path)
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
        embedding=embedding_model
    )
    # 向量数据库本地化存储
    vector_store.save_local(file_path)
    print('faiss数据库本地化保存成功！')
else:
    vector_store = FAISS.load_local(
        file_path, 
        embeddings=embedding_model, 
        allow_dangerous_deserialization=True
    )
    print('加载faiss数据库本地化记录成功！')
# 构建检索器
retriever = vector_store.as_retriever(search_kwargs={"k": 2})
retriever_tool = create_retriever_tool(
    retriever, "book_retriever",
    description="人工智能从“人类数据时代”向“经验时代”的范式转变，并详细描绘了经验时代的特征、技术基础、潜在影响和挑战。"
)

# 工具列表
tools = [search,retriever_tool]

# 构建agent
agent = create_tool_calling_agent(
    llm=model, prompt=prompt, tools=tools)

# 添加工具使用监控
from langchain_core.callbacks import BaseCallbackHandler

class ToolUsageLogger(BaseCallbackHandler):
    def on_tool_start(self, serialized, input_str, **kwargs):
        """当工具开始执行时触发"""
        print(f"\n[TOOL USED] {serialized['name']} with input: {input_str}")
        
executor = AgentExecutor(
    agent=agent, 
    tools=tools, 
    verbose=True,
    return_intermediate_steps=True,  # 返回中间步骤
    callbacks=[ToolUsageLogger()]    # 添加回调
)

# 运行agent 
# msgs = executor.invoke({"input":"北京明天天气如何"})
msgs = executor.invoke({"input":"围绕这篇《欢迎来到经验时代》，说明经验时代可能引发哪些社会风险？有何缓解机制？​​"})

print(msgs['output'])