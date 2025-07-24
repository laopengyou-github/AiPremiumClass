from graphrag import GraphRAG, Document, DocumentStore, VectorStore
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 1. 初始化 GraphRAG 实例
rag = GraphRAG(
    document_store=DocumentStore(),  # 文档存储
    vector_store=VectorStore(embedding=OpenAIEmbeddings()),  # 向量存储
)

# 2. 加载文档（从目录或其他来源）
def load_documents_from_directory(directory: str):
    """从目录加载文档并返回 Document 对象列表"""
    import os

    documents = []
    for filename in os.listdir(directory):
        if filename.endswith('.txt') or filename.endswith('.md'):
            filepath = os.path.join(directory, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()

            # 创建 Document 对象，包含内容和元数据
            doc = Document(
                content=content,
                metadata={
                    'filename': filename,
                    'source': directory,
                }
            )
            documents.append(doc)

    return documents

# 3. 构建索引
def build_index(rag: GraphRAG, documents: list[Document], root_dir: str):
    """构建文档索引"""
    # 文本分割器
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    # 分割文档
    split_docs = text_splitter.split_documents(documents)

    # 添加到文档存储
    rag.document_store.add_documents(split_docs)

    # 构建向量索引
    rag.vector_store.add_documents(split_docs)

    # 构建图结构（识别实体和关系）
    rag.build_graph()

    # 保存索引到磁盘
    rag.save(root_dir)

    return rag

# 4. 执行查询,改一下条目数目
def query_index(rag: GraphRAG, question: str, k: int = 3):
    """执行查询并返回相关文档"""
    # 基于图结构检索
    results = rag.graph_retriever.get_relevant_documents(question, k=k)

    # 或者使用向量检索
    # results = rag.vector_store.similarity_search(question, k=k)

    return results

# 主程序
if __name__ == "__main__":
    # 设置根目录
    root_dir = "ragtest"

    # 加载文档
    documents = load_documents_from_directory(f"{root_dir}/documents")

    # 构建索引
    rag = build_index(rag, documents, root_dir)

    # 执行查询
    question = "介绍一下孙悟空"
    results = query_index(rag, question)

    # 打印结果
    print(f"查询问题: {question}")
    for i, doc in enumerate(results):
        print(f" {doc.page_content[:200]}...")