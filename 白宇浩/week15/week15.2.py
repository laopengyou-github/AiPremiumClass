import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pathlib import Path
from langchain_community.document_loaders import PDFMinerLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough



# 1. 文件路径处理（兼容中文路径）
pdf_path = Path("E:/XUEXIZILIAO/作业/week15/Welcome_to_the_Era_of_Experience.pdf")
if not pdf_path.exists():
    raise FileNotFoundError(f"PDF文件未找到: {pdf_path}")

# 2. 模型初始化
llm = ChatOpenAI(model="glm-4", api_key="your_api_key")
embedding = OpenAIEmbeddings(
    model_name="glm-4-flash-250414",
    api_key=os.environ['api_key'],
        base_url=os.environ['base_url']
)


        
    # 文本分割
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )
    splits = splitter.split_documents(docs)
    
    # 向量存储
    vectorstore = FAISS.from_documents(
        documents=splits,
        embedding=embedding
    )
    vectorstore.save_local('vector_db')
    
except Exception as e:
    print(f"处理失败: {str(e)}")
    raise

# 4. 查询示例
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | StrOutputParser()
)

result = chain.invoke("What are the key characteristics of the era of experience?")
print(result)