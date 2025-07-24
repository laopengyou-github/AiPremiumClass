"""
基于LangChain的RAG（Retrieval-Augmented Generation）执行代码。
完成外部文档导入并进行RAG检索的过程。
外部PDF文档：https://storage.googleapis.com/deepmind-media/Era-of-Experience%20/The%20Era%20of%20Experience%20Paper.pdf
使用 langchain_community.document_loaders.PDFMinerLoader 加载 PDF 文件。
docs = PDFMinerLoader(path).load()
"""

import os
from dotenv import load_dotenv,find_dotenv
import numpy as np

from langchain_community.document_loaders import WebBaseLoader,PDFMinerLoader
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatTongyi
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain import hub
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

def print_vectorstore_details(vectorstore, max_docs=5, max_chars=200, max_vector_elements=10):
    """打印向量数据库中的文档原文和向量信息
    Args:
        vectorstore: FAISS向量数据库实例
        max_docs: 最多打印的文档数
        max_chars: 每个文档最多打印的字符数
        max_vector_elements: 每个向量最多打印的元素数
    """
    index = vectorstore.index
    index_to_docstore = vectorstore.index_to_docstore_id # 获取索引到文档存储的映射
    documents = vectorstore.docstore._dict
    print(f"向量数据库状态:")
    print(f"- 文档数量: {len(documents)}")
    print(f"- 向量数量: {index.ntotal}")
    print(f"- 向量维度: {index.d}")
    
    # 遍历向量并检索对应文档
    for i in range(min(max_docs, index.ntotal)):
        # 从FAISS索引中获取向量
        vector = index.reconstruct(i)
        # 获取文档ID，并打印文档内容
        doc_id = index_to_docstore[i]
        doc = documents.get(doc_id, None)
        print(f"\nQuesion文本内容\n #{doc}， 文档ID: {doc_id}， 向量索引: {i}，向量：{vector[:max_vector_elements]}")
        
        # 使用向量检索对应文档（新版本LangChain推荐方式）
        docs = vectorstore.similarity_search_by_vector(
            embedding=vector,
            k=2  # 只获取最相似的文档
        )
        
        if docs:
            doc = docs[0]
            # 打印文档信息
            print(f"\n文档 #{i}:")
            #print(f"  相似度得分: {score:.4f}")
            print(f"  原文片段: {doc.page_content[:max_chars]}{'...' if len(doc.page_content) > max_chars else ''}")
            print(f"  元数据: {doc.metadata}")
            
            # 打印向量信息
            print(f"  向量维度: {len(vector)}")
            print(f"  向量前{max_vector_elements}个元素: {vector[:max_vector_elements]}...")
            print(f"  向量范数: {np.linalg.norm(vector):.4f}") # 计算向量的L2范数，也就是向量的长度，因为经过L2归一化，所以应该接近1
        else:
            print(f"\n文档 #{i}: 未找到匹配的文档（可能向量数据库已损坏）")

def load_weburl_and_index(web_url, embedding_model):
    """加载网页内容并创建FAISS索引"""
    if not os.path.exists("faiss_index_weburl"):        
        # Load, chunk and index the contents of the blog.
        # 使用WebBaseLoader加载网页内容
        user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
        try:
            # 验证URL格式
            from urllib.parse import urlparse
            parsed_url = urlparse(web_url)
            if not all([parsed_url.scheme, parsed_url.netloc]):
                raise ValueError(f"无效的URL: {web_url}")
            
            loader = WebBaseLoader(
                web_paths=[web_url],  # 使用列表格式
                requests_kwargs={"headers": {"User-Agent": user_agent}}
            )
            
            documents = loader.load()  # 加载文档
            print(f"成功加载 {len(documents)} 个文档")
            print(f"第一个文档长度: {len(documents[0].page_content)}")
            
        except Exception as e:
            print(f"加载文档时出错: {e}")
            import sys
            sys.exit(1)

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200) # 创建文本分割器,块大小为1000，重叠为200
        splits = text_splitter.split_documents(documents) # 分割文档
        print(f"Loaded {len(splits)} splits") # 打印分割后的文档数量

        # vectorstore = FAISS.from_documents(splits,embedding=OpenAIEmbeddings())# 使用FAISS向量存储分割后的文档，用OpenAI的嵌入模型要求api_key环境变量
        # 记录加载embedding模型的时间
        import time
        start_time = time.time() # 记录开始时间
        embedding = embedding_model 
        print(f"加载嵌入模型耗时: {time.time() - start_time:.2f}秒") # 打印加载时间
        # 加载嵌入模型耗时: 129.01秒
        start_time_2 = time.time() # 记录开始时间
        vectorstore = FAISS.from_documents(splits,embedding=embedding) # 使用FAISS向量存储分割后的文档，用HuggingFace的嵌入模型
        print(f"FAISS向量存储耗时：{time.time() - start_time_2:.2f}秒") 
        # 向量存储耗时：0.46秒
        
        # 如果需要，可以将向量存储保存到磁盘
        vectorstore.save_local("faiss_index_weburl") # 保存到本地目录  
    else:
        # 如果向量存储已存在，直接加载
        vectorstore = FAISS.load_local(
            "faiss_index_weburl", 
            embeddings=embedding_model,
            allow_dangerous_deserialization=True
        )
        print(f"从本地加载向量存储，文档数量: {len(vectorstore.docstore._dict)}")
    return vectorstore

def load_pdf_and_index(pdf_path, embedding_model):
    """加载PDF文档并创建FAISS索引"""
    if not os.path.exists("faiss_index_pdf"):
        # 加载PDF文档
        loader = PDFMinerLoader(pdf_path)
        documents = loader.load()
        print(f"成功加载 {len(documents)} 个文档")
        
        # 分割文档
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(documents)
        print(f"分割后文档数量: {len(splits)}")
        
        # 创建FAISS向量存储
        vectorstore = FAISS.from_documents(splits, embedding=embedding_model)
        vectorstore.save_local("faiss_index_pdf")  # 保存到本地目录
        print(f"向量存储已保存到 'faiss_index_pdf' 目录")    
    else:
        # 如果向量存储已存在，直接加载
        vectorstore = FAISS.load_local(
            "faiss_index_pdf", 
            embeddings=embedding_model,  # 使用相同的嵌入模型加载
            allow_dangerous_deserialization=True  # 允许不安全的反序列化
        )
        print(f"从本地加载向量存储，文档数量: {len(vectorstore.docstore._dict)}")
    
    return vectorstore

if __name__ == "__main__":
    # 加载环境变量
    load_dotenv(find_dotenv())

    # 通义千问大模型
    llm = ChatTongyi(
        model="qwen-max",  # 选择模型版本，如 "qwen-long" 或 "qwen-turbo" -- coder模型不行
        top_p=0.8,          # 参数调整，如 top-p 控制生成的多样性
        temperature=0.1,    # 温度参数，控制生成的随机性
        api_key=os.getenv("api_key")
    )
    # 使用HuggingFace的嵌入模型，all-MiniLM-L6-v2是一个bi-encoder模型，适合文本嵌入
    embedding_model = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    ) 

    # 加载网页并创建向量存储
    # vectorstore = load_weburl_and_index(
    #     web_url="https://hawstein.com/2014/03/06/make-thiner-make-friend-with-time/",
    #     embedding_model=embedding_model # 使用HuggingFace的嵌入模型
    # )

    # 加载PDF文档并创建向量存储
    vectorstore = load_pdf_and_index(
        pdf_path="The Era of Experience Paper.pdf", # PDF文档路径
        embedding_model=embedding_model # 使用HuggingFace的嵌入模型
    )
    
    # 打印向量存储的基本信息
    # print_vectorstore_details(vectorstore) # 打印向量存储的基本信息

    # Retriever 构建检索器
    retriever = vectorstore.as_retriever() # 创建检索器

    # prompt 提示词模板
    prompt = hub.pull("rlm/rag-prompt") # 从Hub中获取RAG提示模板

    def format_docs(docs):
        """格式化文档为字符串"""
        return "\n\n".join(doc.page_content for doc in docs)

    # 创建RAG链
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm 
        | StrOutputParser() 
    )

    # 执行查询
    try:
        question = "什么是人工智能的体验时代?"
        answer = rag_chain.invoke(question)
        print(f"\n问题: {question}")
        print(f"回答:\n{answer}")
    except Exception as e:
        print(f"执行RAG链时出错: {e}")

    # 执行结果：
    # 问题: 什么是人工智能的体验时代?
    # 回答:
    # 人工智能的体验时代指的是AI系统能够根据用户的实际体验和反馈来调整其行为和目标，从而更好地满足用户的需求。
    # 在这个时代，AI不仅能够执行特定任务，还能通过与用户的 互动不断学习和优化，以实现更个性化、更自然的服务。
    # 例如，AI可以根据用户的健康数据帮助改善其健身计划，或者根据学习进度帮助用户掌握新语言。
