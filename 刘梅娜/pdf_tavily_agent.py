import os
from typing import List
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain.agents import create_openai_functions_agent
from langchain.agents import AgentExecutor
from langchain_tavily import TavilySearch
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings  # 更新导入

# 设置环境变量以解决 tokenizers 警告
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 加载环境变量
load_dotenv()

class PDFTavilyAgent:
    def __init__(self, pdf_path: str):
        """初始化代理
        Args:
            pdf_path (str): PDF文件路径
        """
        self.pdf_path = pdf_path
        self.setup_tools()
        
    def setup_tools(self):
        """设置必要的工具和模型"""
        try:
            # 初始化PDF加载器
            loader = PyPDFLoader(self.pdf_path)
            pages = loader.load()
            
            # 文本分割
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            splits = text_splitter.split_documents(pages)
            
            # 创建 embeddings 对象，使用本地模型
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'}  # 明确指定使用 CPU
            )
            
            # 创建向量存储
            self.vectorstore = Chroma.from_documents(
                documents=splits,
                embedding=embeddings
            )
            
            # 设置检索器
            self.retriever = self.vectorstore.as_retriever()
            
            # 设置Tavily搜索工具
            self.search = TavilySearch()
            
            # 设置工具列表
            self.tools = [self.search]
            
            # 设置LLM
            self.llm = ChatOpenAI(
                temperature=0,
                model="gpt-3.5-turbo",
                api_key=os.environ['api_key'],
                base_url=os.environ['base_url']
            )
            
            # 创建代理
            prompt = ChatPromptTemplate.from_messages([
                ("system", """你是一个助手，可以:
                1. 从PDF文档中检索相关信息
                2. 使用Tavily搜索互联网获取补充信息
                3. 结合两种信息源回答问题
                
                请确保回答准确、相关且有帮助。"""),
                ("user", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad")
            ])
            
            self.agent = create_openai_functions_agent(
                llm=self.llm,
                tools=self.tools,
                prompt=prompt
            )
            
            self.agent_executor = AgentExecutor(
                agent=self.agent,
                tools=self.tools,
                verbose=True
            )
            
        except Exception as e:
            print(f"设置工具时出错: {str(e)}")
            raise

    def query(self, question: str) -> str:
        """查询功能
        Args:
            question (str): 用户问题
        Returns:
            str: 回答
        """
        try:
            # 从PDF中检索相关内容
            relevant_docs = self.retriever.get_relevant_documents(question)
            context = "\n".join([doc.page_content for doc in relevant_docs])
            
            # 结合PDF内容和搜索功能回答问题
            response = self.agent_executor.invoke({
                "input": f"""基于以下PDF内容和需要时的网络搜索回答问题:
                
                PDF内容:
                {context}
                
                问题: {question}"""
            })
            
            return response["output"]
        except Exception as e:
            print(f"查询时出错: {str(e)}")
            return f"发生错误: {str(e)}"

def main():
    try:
        # 使用示例
        pdf_path = "GPT系列模型搭建、训练及优化.pdf"  # 替换为实际的PDF路径
        agent = PDFTavilyAgent(pdf_path)
        
        # 示例查询
        question = "GPT模型的主要架构是什么？"
        answer = agent.query(question)
        print(f"问题: {question}")
        print(f"回答: {answer}")
    except Exception as e:
        print(f"运行时出错: {str(e)}")

if __name__ == "__main__":
    main()