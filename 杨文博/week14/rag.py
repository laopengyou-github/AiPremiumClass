from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings


class EnhancedLibrarySystem:
    def __init__(self):
        # 1. 知识库构建
        self.books = [
            {"title": "三体", "author": "刘慈欣", "genre": "科幻",
             "desc": "讲述地球文明与三体文明的宇宙博弈，涉及黑暗森林理论"},
            {"title": "活着", "author": "余华", "genre": "文学",
             "desc": "通过农民福贵的一生展现中国近代史变迁"}
        ]

        # 2. 创建语义搜索库
        self.embeddings = HuggingFaceEmbeddings(model_name="shibing624/text2vec-base-chinese")
        self.db = FAISS.from_texts(
            [f"{b['title']}：{b['desc']}" for b in self.books],
            self.embeddings
        )

        # 3. 对话链配置
        self.prompt = PromptTemplate(
            input_variables=["query", "context"],
            template="""
根据以下书籍信息回答问题：
{context}

用户问题：{query}
请按以下规则响应：
1. 借阅咨询：说明书籍内容和适合人群
2. 推荐请求：推荐3本相关书籍并解释理由
3. 其他问题：友好引导至图书管理员"""
        )

        self.llm = Ollama(model="qwen:7b")  # 中文表现更好的模型

    def search_books(self, query):
        # 语义搜索相关书籍
        docs = self.db.similarity_search(query, k=3)
        return "\n".join(d.page_content for d in docs)

    def consult(self, question):
        context = self.search_books(question)
        chain = LLMChain(llm=self.llm, prompt=self.prompt)
        return chain.run(query=question, context=context)


# 使用示例
if __name__ == "__main__":
    library = EnhancedLibrarySystem()

    questions = [
        "我想找关于宇宙的小说",
        "有哪些描写中国历史的书？",
        "如何办理借书证？"
    ]

    for q in questions:
        print(f"\n用户问：{q}")
        print("AI回复：", library.consult(q))
