1. 根据课堂RAG示例，完成外部文档导入并进行RAG检索的过程。
外部PDF文档：https://storage.googleapis.com/deepmind-media/Era-of-Experience%20/The%20Era%20of%20Experience%20Paper.pdf

# 使用 langchain_community.document_loaders.PDFMinerLoader 加载 PDF 文件。
docs = PDFMinerLoader(path).load()

2. 使用graphrag构建一篇小说（自主选择文档）的RAG知识图，实现本地和全局问题的问答。（截图代码运行结果）