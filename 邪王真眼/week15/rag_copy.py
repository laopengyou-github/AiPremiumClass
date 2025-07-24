import os

from langchain import hub
from langchain.vectorstores import Chroma
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers.string import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain


llm = ChatOpenAI(
    model="gpt-4o-mini",
    api_key=os.environ['api_key'],
    base_url=os.environ['base_url']
)

embeddings = OpenAIEmbeddings(
    api_key=os.environ['api_key'],
    base_url=os.environ.get('base_url')
)

loader = WebBaseLoader(
    web_paths=("https://hawstein.com/2014/03/06/make-thiner-make-friend-with-time/",),
)

docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)


## rag1
vectorstore1 = FAISS.from_documents(documents=splits, embedding=embeddings)

retriever1 = vectorstore1.as_retriever()
prompt = hub.pull("rlm/rag-prompt")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain1 = (
 {"context": retriever1 | format_docs, "question": RunnablePassthrough()}
 | prompt
 | llm
 | StrOutputParser()
)

response1 = rag_chain1.invoke("任务分解的⽅法有哪些?")
print(response1)


## rag2
vectorstore2 = Chroma.from_documents(documents=splits, embedding=embeddings)
retriever2 = vectorstore2.as_retriever()
system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain2 = create_retrieval_chain(retriever2, question_answer_chain)

response2 = rag_chain2.invoke({"input": "任务分解的⽅法有哪些?"})

print(response2["answer"])
