#加载配置
from dotenv import load_dotenv,find_dotenv 
load_dotenv(find_dotenv())

#构建LLM
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import os
model = ChatOpenAI(
    model="glm-4-flash-250414",
    base_url=os.environ["base_url"],
    api_key=os.environ["api_key"]
)

#prompt
from langchain import hub
prompt = hub.pull("hwchase17/openai-functions-agent")

#工具
 #tavily搜索
from langchain_community.tools.tavily_search import TavilySearchResults
tavily_search = TavilySearchResults(max_results=2)

 #本地FAISS
from langchain_community.vectorstores import FAISS
embedding_model = OpenAIEmbeddings(
    model="embedding-3",
    base_url=os.environ["base_url"],
    api_key=os.environ["api_key"]
)
if os.path.exists("hd_doc"):
    vector_store = FAISS.load_local(
        folder_path="hd_doc",
        embeddings=embedding_model,
        allow_dangerous_deserialization=True
    )
    print('加载faiss数据库本地化记录成功！')
else:
    #1.加载文档 2.splitter 3.生成向量（vectorstore）并保存
    from langchain_community.document_loaders import PDFMinerLoader
    loader = PDFMinerLoader(file_path="D:/study/ai/code/week16/hd.pdf")
    doc = loader.load()
    
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n','\n',''],
        chunk_size=1000,
        chunk_overlap=100
    )
    doc_sp = splitter.split_documents(doc)

    vector_store = FAISS.from_documents(
        documents=doc_sp,
        embedding=embedding_model
    )
    vector_store.save_local("hd_doc")
    print('faiss数据库本地化保存成功！')

#构建检索器
from langchain.tools.retriever import create_retriever_tool
retriever = vector_store.as_retriever(searrch_kwargs={"k":2})
retriever_tool = create_retriever_tool(
    retriever=retriever,
    name="hd公司制度",
    description="一份hd公司关于总部员工履职待遇、业务支出管理办法，所有有关“hd公司”的内容都可以使用这个工具进行检索，所有出现“hd”关键字的问题，使用这个工具检索。"
)

#构建agent
from langchain.agents import create_tool_calling_agent,AgentExecutor
agent = create_tool_calling_agent(
    llm=model,
    tools=[tavily_search,retriever_tool],
    prompt=prompt
)

#executor
executor = AgentExecutor(
    agent=agent,
    tools=[tavily_search,retriever_tool],
    verbose=True
)

#invoke
msgs = executor.invoke({"input":"hd公司关于总部员工履职待遇、业务支出管理办法中，如何认定过错行为？"})
print(msgs['output'])
