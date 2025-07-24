# -*- coding: utf-8 -*-
"""
使用LangChain Agent实现PDF和TAVILY工具组合动态调用
"""

import os
import requests
from langchain_community.document_loaders import PDFMinerLoader
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.agents import AgentExecutor, create_react_agent
from langchain import hub
from langchain.tools import Tool
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI

def setup_pdf_tavily_agent():
    os.environ["TAVILY_API_KEY"] = "安全考虑这里去掉"  # 请在https://app.tavily.com获取
    os.environ["OPENAI_API_KEY"] = "安全考虑这里去掉"  # 请在https://platform.openai.com获取
    
    def load_and_process_pdf(pdf_url):
        """下载并处理PDF文档"""
        local_path = "temp_pdf.pdf"
        response = requests.get(pdf_url)
        with open(local_path, 'wb') as f:
            f.write(response.content)

        loader = PDFMinerLoader(local_path)
        documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", "。", "！", "？", "；"]
        )
        texts = text_splitter.split_documents(documents)

        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
        db = FAISS.from_documents(texts, embeddings)
        return db

    pdf_url = "https://storage.googleapis.com/deepmind-media/Era-of-Experience%20/The%20Era%20of%20Experience%20Paper.pdf"
    pdf_db = load_and_process_pdf(pdf_url)

    tools = [

        TavilySearchResults(),

        Tool(
            name="pdf_search",
            func=lambda query: str(pdf_db.similarity_search(query, k=3)),
            description="当需要回答关于DeepMind《经验时代》PDF文档内容的问题时使用"
        )
    ]

    prompt = hub.pull("hwchase17/react-chat")
    
    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0,
        streaming=True
    )

    agent = create_react_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        max_iterations=5,
        handle_parsing_errors=True
    )

    test_questions = [
        "PDF中提到的'经验时代'是什么意思？",
        "结合网络最新信息，DeepMind在AI安全方面有哪些研究进展？",
        "根据PDF内容，说明DeepMind对AGI发展的看法",
        "搜索并总结PDF中关于强化学习的最新应用案例"
    ]

    print("="*60)
    print("PDF和Tavily动态调用测试结果")
    print("="*60)
    
    for question in test_questions:
        print(f"\n问题: {question}")
        try:
            result = agent_executor.invoke({
                "input": question,
                "chat_history": []
            })
            print("回答:", result["output"])
            print("使用的工具:", [tool.name for tool in result.intermediate_steps])
        except Exception as e:
            print(f"执行出错: {str(e)}")

if __name__ == "__main__":
    setup_pdf_tavily_agent()
