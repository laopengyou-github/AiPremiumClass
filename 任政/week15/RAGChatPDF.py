from langchain_community.document_loaders import PDFMinerLoader
from langchain_huggingface import HuggingFaceEndpoint , HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain import hub
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv, find_dotenv
import os
import requests
import tempfile
import os


if __name__ == '__main__':
    # 从 .env 文件加载环境变量
    load_dotenv()
    # ========== 1. 下载并加载PDF文档 ==========
    pdf_url = "https://storage.googleapis.com/deepmind-media/Era-of-Experience%20/The%20Era%20of%20Experience%20Paper.pdf"

    # 使用PDFMinerLoader加载PDF内容
    loader = PDFMinerLoader(pdf_url)
    docs = loader.load()

    # 检查加载结果
    print(f"加载文档页数: {len(docs)}")
    print(f"第一页内容片段: {docs[0].page_content[:100]}...")

    # ========== 2. 初始化DeepSeek模型 ==========
    # DeepSeek生成模型 (用于回答问题)
    llm = ChatOpenAI(
        model="deepseek-chat",
        api_key=os.getenv("API_KEY"),  # 使用.env中的API_KEY
        base_url=os.getenv("BASE_URL")  # 使用.env中的BASE_URL
    )
    # DeepSeek嵌入模型 (用于向量化文本)
    embedding_model = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-zh-v1.5",
        model_kwargs={"device": "cpu"},  # 使用GPU可改为"cuda"
        encode_kwargs={
            "normalize_embeddings": True,  # 归一化向量
            # "show_progress_bar": True
        }
    )

    # ========== 3. 文档分割 ==========
    # 创建文本分割器（针对PDF内容优化）
    splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '. ', '? ', '! ', '; ', '。', '？', '！'],  # 中英文分割符
        chunk_size=500,  # 每个块的最大字符数
        chunk_overlap=50,  # 块间重叠字符数
        length_function=len  # 使用字符长度而非token计数
    )

    splited_docs = splitter.split_documents(docs)
    print(f"分割后文档块数量: {len(splited_docs)}")

    # ========== 4. 创建和加载向量数据库 ==========
    db_path = 'deepseek_faiss_db'

    if not os.path.exists(db_path):
        # 创建向量数据库
        vector_store = FAISS.from_documents(
            documents=splited_docs,
            embedding=embedding_model
        )

        # 向量数据库本地化存储
        vector_store.save_local(db_path)
        print('FAISS 数据库本地化保存成功！')
    else:
        # 加载现有数据库
        vector_store = FAISS.load_local(
            db_path,
            embeddings=embedding_model,
            allow_dangerous_deserialization=True
        )
        print('加载 FAISS 数据库本地化记录成功！')

    # ========== 5. 构建检索器 ==========
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})  # 返回top3相关文档


    # 格式化检索到的文档
    def format_docs(docs):
        """将检索到的文档块格式化为字符串"""
        return "\n\n".join([f"内容: {doc.page_content}"for doc in docs])


    # ========== 6. 构建DeepSeek优化的提示模板 ==========
    # DeepSeek专用提示模板（使用原生对话格式）
    deepseek_prompt_template = """
    <|im_start|>system
    你是一个专业的研究助手，请基于以下上下文回答问题。
    如果不知道答案，请说"我不知道"，不要编造答案。<|im_end|>

    <|im_start|>context
    {context}<|im_end|>

    <|im_start|>user
    {question}<|im_end|>

    <|im_start|>assistant
    """

    # 从Hub中拉取提示模板（或使用自定义模板）
    try:
        # 尝试使用LangChain Hub的RAG提示
        prompt = hub.pull('rlm/rag-prompt')
        print("使用LangChain Hub的标准RAG提示模板")
    except:
        # 使用DeepSeek优化的自定义提示
        from langchain_core.prompts import PromptTemplate

        prompt = PromptTemplate.from_template(deepseek_prompt_template)
        print("使用DeepSeek优化的自定义提示模板")

    # ========== 7. 构建RAG链 ==========
    rag_chain = (
            {"context": retriever | format_docs,  # 检索并格式化文档
             "question": RunnablePassthrough()}  # 直接传递问题
            | prompt  # 应用提示模板
            | llm  # 使用DeepSeek生成答案
            | StrOutputParser()  # 解析为字符串输出
    )

    # ========== 8. 执行RAG检索 ==========
    questions = [
        "What is the main theme of the Era of Experience paper?",
        "How does the paper define 'experience' in the context of AI?",
        "作者在论文中提到的'体验经济'的核心观点是什么？"  # 测试中文问题
    ]

    for question in questions:
        print(f"\n{'=' * 50}\n[问题] {question}\n{'=' * 50}")

        # 执行RAG查询
        response = rag_chain.invoke(question)

        # 打印答案
        print(f"\n[答案] {response}\n")