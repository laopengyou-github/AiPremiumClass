from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain import hub
from langchain_community.vectorstores import FAISS
from langchain_tavily import TavilySearch
from langchain.tools.retriever import create_retriever_tool
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_community.document_loaders import PDFMinerLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.embeddings import Embeddings
import requests


load_dotenv()
api_key = os.getenv("api_key")
base_url = os.getenv("base_url")
PDF_PATH = os.getenv("PDF_PATH")

class DoubaoEmbedding(Embeddings):
    def __init__(self, base_url, api_key):
        self.base_url = base_url
        self.api_key = api_key

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        headers = {"Authorization": f"Bearer {self.api_key}"}
        payload = {"input": texts, "model": "doubao-embedding-text-240715"}
        resp = requests.post(
            f"{self.base_url}/embeddings",
            json=payload,
            headers=headers
        )
        return [d["embedding"] for d in resp.json()["data"]]

    def embed_query(self, text: str) -> list[float]:
        headers = {"Authorization": f"Bearer {self.api_key}"}
        payload = {"input": [text], "model": "doubao-embedding-text-240715"}
        resp = requests.post(
            f"{self.base_url}/embeddings",
            json=payload,
            headers=headers
        )
        return resp.json()["data"][0]["embedding"]


llm = ChatOpenAI(
    model="doubao-seed-1.6-250615",
    api_key=os.environ['api_key'],  
    base_url=os.environ['base_url']  
)

embedding_model = DoubaoEmbedding(
    base_url=os.environ["base_url"],
    api_key=os.environ["api_key"]
)


prompt = hub.pull("hwchase17/openai-functions-agent")
search_tool = TavilySearch(max_results=5, tavily_api_key=os.getenv("TAVILY_API_KEY"))

# 构建向量库
if not os.path.exists("local_save"):
    os.makedirs("local_save")

loader = PDFMinerLoader(PDF_PATH)
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(documents)

vector_store = FAISS.from_documents(splits, embedding_model)
vector_store.save_local("local_save")
print("向量库创建完成")


vector_store = FAISS.load_local(
    "local_save", 
    embeddings=embedding_model,
    allow_dangerous_deserialization=True
)

retriever = vector_store.as_retriever(search_kwargs={"k": 7})

retriever_tool = create_retriever_tool(
    retriever,
    name="book_retriever",
    description="一款基于用户查询，从论文中检索相关信息的工具。能够精准定位书中最贴切的段落，为用户提供所需知识，总结回答问题。"
)


tools = [search_tool, retriever_tool]


agent = create_tool_calling_agent(
    llm=llm,
    prompt=prompt,
    tools=tools
)


executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,  
    handle_parsing_errors=True
)


def run_agent(query):
    response = executor.invoke({
        "input": query
    })
    return response


while True:
    query = input("问题：")
    if query.strip().lower() == 'q':
        break
    response = run_agent(query)
    print("回答：", response['output'])


