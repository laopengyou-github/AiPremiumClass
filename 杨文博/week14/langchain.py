from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain_community.llms import Ollama  # 或使用ChatOpenAI


# 1. 初始化主题约束的聊天系统
class ThemedChatSystem:
    def __init__(self, theme="人工智能"):
        self.theme = theme
        self.llm = Ollama(model="llama3")  # 本地模型
        # 如需使用OpenAI：ChatOpenAI(model="gpt-3.5-turbo")

        self.memory = ConversationBufferMemory()
        self.chain = ConversationChain(
            llm=self.llm,
            memory=self.memory,
            prompt=self._build_prompt()
        )

    def _build_prompt(self):
        from langchain.prompts import PromptTemplate
        return PromptTemplate(
            input_variables=["history", "input"],
            template=f"""你是一个专注讨论{self.theme}领域的AI助手，请严格遵守：
1. 当用户询问非{self.theme}话题时，礼貌拒绝回答
2. 保持专业但友好的语气
3. 对复杂概念用比喻解释

当前对话历史：
{{history}}

用户新输入：{{input}}"""
        )

    def chat(self, message):
        response = self.chain.run(input=message)
        # 记录对话到内存（自动完成）
        return response


# 使用示例
if __name__ == "__main__":
    theme = "量子物理"
    bot = ThemedChatSystem(theme)

    print(f"已启动{theme}主题聊天（输入q退出）")
    while True:
        user_input = input("用户: ")
        if user_input == "q":
            break
        print("AI:", bot.chat(user_input))
