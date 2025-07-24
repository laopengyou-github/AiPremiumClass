 ##llm调用封装
from dotenv import load_dotenv,find_dotenv
import os
# from  langchain_community.document_loaders import WebBaseLoader 
from  langchain_community.document_loaders import PDFMinerLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain import hub 
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
if __name__=='__main__':
    load_dotenv(find_dotenv())
  
    ##langsmith
    LANGSMITH_TRACING=True
    LANGSMITH_ENDPOINT="https://api.smith.langchain.com"
    LANGSMITH_API_KEY=os.environ['LANGSMITH_API_KEY']
    LANGSMITH_PROJECT="pr-dear-sandbar-100"
    OPENAI_API_KEY=os.environ['OPEN_API_KEY']

    ##llm model
    llm=ChatOpenAI(  
    model="gpt-4o-mini",
    api_key=os.environ['OPEN_API_KEY'],
    base_url=os.environ['BASE_URL']
)   
    
    ##加载网页中文本内容，转换为langchain处理的document
    # WebBaseLoader(web_path='')
    path=f'https://storage.googleapis.com/deepmind-media/Era-of-Experience%20/The%20Era%20of%20Experience%20Paper.pdf'



    embedding_model=OpenAIEmbeddings(
            openai_api_key=os.environ['OPEN_API_KEY'],
            base_url=os.environ['BASE_URL']
    )
    
    if not os.path.exists('local_save'):
        docs = PDFMinerLoader(path).load()
        ##textsplitter，实现加载后document分割
        # ##基于长度拆分，以下是默认规则，可以省略的参数
        splitter=RecursiveCharacterTextSplitter(
         separators=['\n\n','\n',''], ##\n\n 段落，\n 句子，
         chunk_size=1000,
         chunk_overlap=100 ##切分时覆盖的长度
    )

        splited_docs=splitter.split_documents(docs)


        ##创建向量数据库（内存中）对chunk进行向量化和存储

        vector_store=FAISS.from_documents(
            documents=splited_docs,
            embedding=embedding_model)
        

        vector_store.save_local('local_save')
        print('faiss数据库本地化保存成功')

        # FAISS.load_local()
        # print(splited_docs)
    else:
        vector_store=FAISS.load_local(
            'local_save',
            embeddings=embedding_model,
            allow_dangerous_deserialization=True
        )
        print('加载faiss数据库本地化记录成功')

    ##构建检索器 
    retriever=vector_store.as_retriever(search_kwwargs={"k":2})

    res=retriever.invoke("经验时代的智能体与人类数据时代的 LLM\（大型语言模型\）在 “学习方式” 上有哪些区别")
    def format_docs(docs):
        return '\n\n'.join([doc.page_content for doc in res])
    # prompt=f"""
    # you are  helpful assist can use content answer user question:

    # """
    ##prompt
    prompt=hub.pull("rlm/rag-prompt")
    
    
    ##构建rag chain  RunnablePassthrough()占位符
    rag_chain=(
        {"context":retriever|format_docs,"question":RunnablePassthrough()}
        |prompt
        |llm 
        |StrOutputParser()
    )
    
    #rag检索
    response=rag_chain.invoke("经验时代的智能体与人类数据时代的 LLM\（大型语言模型\）在 “学习方式” 上有哪些区别")
    print("AI response:",response)



