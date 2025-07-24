from langchain_openai import ChatOpenAI
from dotenv import load_dotenv, find_dotenv
from langchain import hub
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.agents import create_tool_calling_agent, AgentExecutor
import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.tools.retriever import create_retriever_tool
from langchain_community.document_loaders import PDFMinerLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv(find_dotenv())

# 设置环境变量（关键配置）
os.environ.update({
    "USER_AGENT": "AcademicResearchBot/1.0",
    "HF_ENDPOINT": "https://hf-mirror.com",  # 使用镜像服务器
    "HF_HUB_DISABLE_SYMLINKS_WARNING": "1",
    "TF_ENABLE_ONEDNN_OPTS": "0",
    "HF_HOME": "./hf_cache"  # 自定义缓存目录
})

# 构建LLM
model = ChatOpenAI(
        model="glm-4-flash-250414",
        base_url=os.environ['base_url'],
        api_key=os.environ['api_key'],
        temperature=0.7
    ) # 支持function call

# 提示词模版
prompt = hub.pull("hwchase17/openai-tools-agent")

# 构建工具
search = TavilySearchResults(tavily_api_key = os.environ['tavily_api_key'], max_results = 2)

# 使用 huggingface 的 bge-small-zh-v1.5 模型进行向量化
embedding_model = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-zh-v1.5",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True},
        cache_folder=os.environ["HF_HOME"]
    )

if not os.path.exists('faiss_index'):
    # 定义pdf路径
    pdf_path = "https://storage.googleapis.com/deepmind-media/" \
    "Era-of-Experience%20/The%20Era%20of%20Experience%20Paper.pdf"

    # 构建文档加载器
    pdf_loader = PDFMinerLoader(pdf_path)
    pdf_docs = pdf_loader.load()

    # 定义文档切分器
    text_splitter = RecursiveCharacterTextSplitter(
        separators = ['\n\n', '\n', ''], # 分隔符优先级从高到低
        chunk_size = 1024,
        chunk_overlap = 128
    )

    # # 切分文档
    splited_docs = text_splitter.split_documents(pdf_docs)

    # 创建向量数据库(内存中)，对chunk进行向量化和存储
    # 构建向量数据库
    vector_store = FAISS.from_documents(documents=splited_docs, embedding=embedding_model)

    # 保存向量数据库
    vector_store.save_local(folder_path="faiss_index")
    print("faiss向量数据库保存成功！")
else:
    # 加载向量存储
    vector_store = FAISS.load_local(
            'faiss_index',
            embeddings=embedding_model,
            allow_dangerous_deserialization=True
    )
    print("faiss向量数据库加载成功！")

# 构建检索器
retriever = vector_store.as_retriever(search_kwargs={"k": 2})

retriever_tool = create_retriever_tool(
    retriever,
    name="pdf_retriever",
    description='''围绕这篇PDF回答问题，必须使用此工具获取原文内容。输入应为具体问题。'''
)

# 工具列表
tools = [search,retriever_tool]

# 构建agent
agent = create_tool_calling_agent(model, tools, prompt)

# agent executor
executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True
    )

# 运行agent
# msgs = executor.invoke({"input":"北京明天天气如何"})
msgs = executor.invoke({"input":"人工智能未来的发展如何"})

print(msgs['output'])

