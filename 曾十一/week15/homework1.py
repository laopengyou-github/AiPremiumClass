# -*- coding: utf-8 -*-

import os
import requests
from dotenv import load_dotenv
# 导入智谱AI的模型和嵌入
from langchain_community.chat_models.zhipuai import ChatZhipuAI
from langchain_community.embeddings.zhipuai import ZhipuAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain import hub

def main():

    load_dotenv()

    llm = ChatZhipuAI(
        model="glm-4-flash",
        api_key=os.environ['ZHIPUAI_API_KEY'],
        temperature=0.1,
    )
    embedding_model = ZhipuAIEmbeddings(
        api_key=os.environ['ZHIPUAI_API_KEY']
    )

    pdf_url = "https://storage.googleapis.com/deepmind-media/Era-of-Experience%20/The%20Era%20of%20Experience%20Paper.pdf"
    local_pdf_path = "The_Era_of_Experience_Paper.pdf"
    faiss_index_path = "faiss_index_pdf_zhipu"

    if not os.path.exists(faiss_index_path):
        if not os.path.exists(local_pdf_path):
            print(f"正在从URL下载PDF: {pdf_url}")
            try:
                response = requests.get(pdf_url, timeout=30)
                response.raise_for_status()
                with open(local_pdf_path, 'wb') as f:
                    f.write(response.content)
                print(f"PDF已成功下载并保存至: {local_pdf_path}")
            except requests.exceptions.RequestException as e:
                print(f"下载PDF失败: {e}")
                return

        print(f"正在加载PDF文档: {local_pdf_path}")
        loader = PyPDFLoader(file_path=local_pdf_path)
        docs = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=['\n\n', '\n', ' ', '']
        )
        split_docs = text_splitter.split_documents(docs)
        print(f"文档已分割成 {len(split_docs)} 个块。")

        original_count = len(split_docs)
        split_docs = [doc for doc in split_docs if doc.page_content.strip()]
        if not split_docs:
            print("错误：过滤后没有有效的文档块可用于创建索引。")
            return
        print(f"过滤掉空块后，剩余 {len(split_docs)} 个块（从 {original_count} 个中）。")


        print("正在使用智谱AI嵌入创建并保存FAISS向量数据库...")
        batch_size = 25  # 智谱AI嵌入API的批处理大小限制
        vector_store = None

        for i in range(0, len(split_docs), batch_size):
            batch = split_docs[i:i + batch_size]
            if not batch:
                continue
            
            total_batches = (len(split_docs) + batch_size - 1) // batch_size
            print(f"正在处理批次 {i//batch_size + 1}/{total_batches}...")
            
            if vector_store is None:
                # 使用第一个批次初始化向量数据库
                vector_store = FAISS.from_documents(
                    documents=batch,
                    embedding=embedding_model
                )
            else:
                # 将后续批次添加到已存在的数据库中
                vector_store.add_documents(documents=batch)

        if vector_store is None:
            print("错误：未能创建向量数据库，没有有效的文档块。")
            return
            
        vector_store.save_local(faiss_index_path)
        print(f"FAISS向量数据库已成功保存至: {faiss_index_path}")
    
    else:
        print(f"正在从本地加载FAISS向量数据库: {faiss_index_path}")
        vector_store = FAISS.load_local(
            faiss_index_path, 
            embeddings=embedding_model, 
            allow_dangerous_deserialization=True
        )
        print("FAISS向量数据库加载成功！")

    # --- 4. RAG链构建与执行 ---
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})
    prompt = hub.pull("rlm/rag-prompt")

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # --- 5. 执行RAG并打印结果 ---
    question = "请总结一下这篇文档的核心观点是什么？"
    print("\n--- 开始RAG检索 (使用智谱AI) ---")
    print(f"问题: {question}")
    
    full_response = ""
    for chunk in rag_chain.stream(question):
        print(chunk, end='', flush=True)
        full_response += chunk
    
    print("\n\n--- RAG检索完成 ---")

if __name__ == '__main__':
    # 在运行此脚本前，请确保已安装所有必要的库:
    # pip install langchain langchain-community langchain-zhipuai python-dotenv faiss-cpu pypdf requests langchainhub
    main()