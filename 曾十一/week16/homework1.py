# main_app_final.py

import os
import time
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import ZhipuAIEmbeddings
from langchain import hub
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain.tools.retriever import create_retriever_tool
from langchain_openai import ChatOpenAI

# --- 1. 配置区域 (请根据您的需求修改这里) ---

# ==> 1.1: 指定包含所有PDF文件的文件夹路径
# 程序会自动扫描这个文件夹下所有扩展名为 .pdf 的文件。
PDF_SOURCE_FOLDER = r"/mnt/data_1/zfy/self/八斗精品班/第十六周_GPT系列模型搭建训练及优化/homework/rag/PDF" 

# ==> 1.2: 指定向量库保存的本地路径
# 这个文件夹将用于存储处理好的PDF知识。
VECTOR_STORE_PATH = r"/mnt/data_1/zfy/self/八斗精品班/第十六周_GPT系列模型搭建训练及优化/homework/rag/vector_store"

# ==> 1.3: 【重要】为您的知识库工具编写一个准确的描述
# 这个描述需要概括您文件夹内所有PDF的核心内容，以便Agent知道何时使用它。
KNOWLEDGE_BASE_DESCRIPTION = "专门用于检索和回答本地知识库中PDF文档的内容。知识库包含关于langchain和RAG的详细资料。当问题与这些特定领域相关时，应优先使用此工具。"


# --- 2. 准备工作：加载环境变量 ---
print("--- 准备工作：加载环境变量 ---")
load_dotenv()
api_key = os.getenv("api_key")
base_url = os.getenv("base_url")

if not api_key or not base_url:
    raise ValueError("请确保 .env 文件中已配置 api_key 和 base_url")

# --- 3. 核心逻辑：加载或构建向量库 ---
print("\n--- 核心逻辑：检查并准备向量库 ---")

# 初始化Embedding模型
embedding_model = ZhipuAIEmbeddings(
    model="embedding-3",
    api_key=api_key,
    base_url=base_url
)

if os.path.exists(VECTOR_STORE_PATH):
    # 如果路径存在，直接加载
    print(f"发现已存在的向量库 '{VECTOR_STORE_PATH}'，正在直接加载...")
    start_time = time.time()
    vector_store = FAISS.load_local(
        VECTOR_STORE_PATH,
        embeddings=embedding_model,
        allow_dangerous_deserialization=True
    )
    load_time = time.time() - start_time
    print(f"向量库加载完成，耗时 {load_time:.2f} 秒。")
else:
    # 如果路径不存在，则执行构建流程
    print(f"未发现向量库 '{VECTOR_STORE_PATH}'，开始进入构建流程...")
    
    # 检查PDF源文件夹是否存在
    if not os.path.exists(PDF_SOURCE_FOLDER) or not os.path.isdir(PDF_SOURCE_FOLDER):
        raise FileNotFoundError(f"错误：PDF源文件夹 '{PDF_SOURCE_FOLDER}' 不存在或不是一个文件夹。请先创建该文件夹并放入PDF文件。")
        
    start_time = time.time()
    
    # 3.1 扫描并加载指定文件夹中的所有PDF文档
    all_documents = []
    print(f"开始扫描文件夹 '{PDF_SOURCE_FOLDER}' 中的PDF文件...")
    
    pdf_files_found = [f for f in os.listdir(PDF_SOURCE_FOLDER) if f.lower().endswith(".pdf")]

    if not pdf_files_found:
         raise FileNotFoundError(f"错误：在文件夹 '{PDF_SOURCE_FOLDER}' 中没有找到任何PDF文件。")

    for pdf_file in pdf_files_found:
        file_path = os.path.join(PDF_SOURCE_FOLDER, pdf_file)
        try:
            print(f"  - 正在加载: {file_path}")
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            all_documents.extend(documents)
            print(f"    '{pdf_file}' 加载完成，包含 {len(documents)} 页。")
        except Exception as e:
            print(f"  - 警告: 加载文件 '{pdf_file}' 时发生错误: {e}，已跳过。")

    if not all_documents:
        raise ValueError("未能成功加载任何PDF文档，请检查PDF文件是否损坏。")

    # 3.2 切分文档
    print("\n正在将所有文档切分成小块...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = text_splitter.split_documents(all_documents)
    print(f"文档切分完成，共得到 {len(split_docs)} 个文本块。")



    BATCH_SIZE = 64
    
    # 3.3.1 先用第一个批次创建向量库
    first_batch = split_docs[:BATCH_SIZE]
    print(f"正在处理第1批，共 {len(first_batch)} 个文本块...")
    vector_store = FAISS.from_documents(first_batch, embedding_model)
    print("初始向量库创建成功。")

    # 3.3.2 循环处理剩余的批次，并添加到已有的向量库中
    total_docs = len(split_docs)
    for i in range(BATCH_SIZE, total_docs, BATCH_SIZE):
        batch_num = (i // BATCH_SIZE) + 1
        remaining_docs = total_docs - i
        current_batch_size = min(BATCH_SIZE, remaining_docs)
        
        print(f"\n正在处理第 {batch_num} 批，共 {current_batch_size} 个文本块...")
        
        current_batch = split_docs[i:i + BATCH_SIZE]
        vector_store.add_documents(current_batch)
        print("本批次已成功添加到向量库。")
    
    print("\n所有批次处理完毕，向量库构建成功！")
    # ===================================================================
    
    # 3.4 保存向量库
    vector_store.save_local(VECTOR_STORE_PATH)
    print(f"向量库已保存至 '{VECTOR_STORE_PATH}'，下次将直接加载。")
    build_time = time.time() - start_time
    print(f"整个构建过程耗时 {build_time:.2f} 秒。")

# --- 4. 构建Agent和工具 (此部分代码无需改动) ---
print("\n--- 构建Agent和工具 ---")
search_tool = DuckDuckGoSearchRun(name="web_search")
retriever = vector_store.as_retriever(search_kwargs={"k": 3})
knowledge_base_tool = create_retriever_tool(
    retriever,
    "local_knowledge_base",
    KNOWLEDGE_BASE_DESCRIPTION
)
tools = [search_tool, knowledge_base_tool]
print("工具列表创建完成: [Tavily Web Search, Local Knowledge Base]")
model = ChatOpenAI(model="glm-4-flash-250414", api_key=api_key, base_url=base_url)
prompt = hub.pull("hwchase17/openai-functions-agent")
agent = create_tool_calling_agent(llm=model, prompt=prompt, tools=tools)
executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
print("Agent构建完成，准备接收指令。")

# --- 5. 运行和演示 (此部分代码无需改动) ---
print("\n" + "="*80)
print("🚀 [演示1] 测试通用知识问题 (Agent应选择 'tavily_search_results_json')")
print("="*80)
#executor.invoke({"input": "日本现在几点了？"})

# Based on the current time information, let's construct a relevant and verifiable query.
# The current time is 3:46 PM JST. A simple query would be "What time is it in Tokyo?".
# The agent should use Tavily search for this.
executor.invoke({"input": "What time is it in Tokyo?"})


print("\n" + "="*80)
print("📚 [演示2] 测试本地知识库问题 (Agent应选择 'local_knowledge_base')")
print("="*80)
executor.invoke({"input": "请根据知识库内容，总结一下langchain和RAG的相关内容。"})