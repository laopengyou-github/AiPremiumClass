from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_community.tools import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI,OpenAIEmbeddings
from langchain import hub
from dotenv import load_dotenv, find_dotenv
from langchain_community.document_loaders import PDFMinerLoader
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.tools.retriever import create_retriever_tool

if __name__ == '__main__':
    load_dotenv(find_dotenv())

    llm = ChatOpenAI(
        model_name='glm-4-flash-250414',
        openai_api_key=os.getenv('api_key'),
        openai_api_base=os.getenv('base_url'),
        temperature=0.7,
    )

    llm_embeddings = OpenAIEmbeddings(
        model='embedding-3',
        openai_api_key=os.getenv('api_key'),
        openai_api_base=os.getenv('base_url'),
    )

    # 构建网络搜索工具
    search = TavilySearchResults(
        tavily_api_key=os.environ['tavily_api_key'],
        max_results=2, #最多返回2个
        sort_by_relevance=True#按照相近的大小进行排序
    )

    # 提示词模版
    system_message = """你是一个智能助手，可以使用以下函数：

    {functions}

    如果你决定调用某个函数，请以下面的 JSON 格式返回：
    {{
      "name": "函数名称",
      "arguments": {{
        "参数1": "值1",
        "参数2": "值2"
      }}
    }}

    如果没有合适的函数，请直接回复用户的问题。"""

    # # 构建 prompt 模板
    # prompt = ChatPromptTemplate.from_messages([
    #     ("system", "你是一个智能助手，可以调用工具来回答问题。"),
    #     MessagesPlaceholder(variable_name="chat_history", optional=True),
    #     ("human", "{input}"),
    #     MessagesPlaceholder(variable_name="agent_scratchpad")
    # ])

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_message),
        MessagesPlaceholder(variable_name="agent_scratchpad", optional=True),
        ("human", "{input}")
    ])
    # prompt = hub.pull("hwchase17/openai-functions-agent")

    if not os.path.exists('local_save'):
        path_url = 'https://storage.googleapis.com/deepmind-media/Era-of-Experience%20/The%20Era%20of%20Experience%20Paper.pdf'
        loader_pdf = PDFMinerLoader(path_url,mode='page').load()

        splitter = RecursiveCharacterTextSplitter(
            separators=['\n\n','\n',' ',''],
            chunk_size=1000,
            chunk_overlap=100,
        )

        splitter_pdf = splitter.split_documents(loader_pdf)

        vectors_stores = FAISS.from_documents(
            documents=splitter_pdf,
            embedding=llm_embeddings,

        )
        vectors_stores.save_local(folder_path='local_save')
        print('保存成功')

    else:
        vectors_stores = FAISS.load_local(
            folder_path='./local_save',
            embeddings=llm_embeddings,
            allow_dangerous_deserialization=True
        )

        print('加载成功')

    retriever = vectors_stores.as_retriever(search_kwargs={'K':2})

    retriever_tool = create_retriever_tool(
        retriever,
        'pdf_str',
        description='AI人工智能体的发展速度的讲解'
    )

    # 工具列表
    tools = [search,retriever_tool]
    # 构建agent
    agent = create_tool_calling_agent(llm=llm, prompt=prompt, tools=tools)

    # agent
    executor = AgentExecutor(agent=agent, tools=tools,verbose=True)
    # 运行agent
    msgs = executor.invoke({"input": "海贼王是什么"})
    print(msgs)
