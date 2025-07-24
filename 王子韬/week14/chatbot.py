from dotenv import find_dotenv, load_dotenv
import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


class TopicChatSystem:
    def __init__(self):
        # 加载环境变量
        load_dotenv(find_dotenv())

        # 初始化LLM模型
        self.model = ChatOpenAI(
            model="glm-4-flash-250414",
            base_url=os.environ['BASE_URL'],
            api_key=os.environ['API_KEY'],
            temperature=0.7
        )

        # 预定义主题及其系统提示词
        self.topics = {
            "1": {
                "name": "科技讨论",
                "prompt": "你是一个科技专家，专门讨论人工智能、编程、新兴技术等话题。请用专业但易懂的语言回答问题，并提供实用的见解。如果用户说退出、再见等词汇，请礼貌地结束对话。"
            },
            "2": {
                "name": "文学交流",
                "prompt": "你是一个文学爱好者，熟悉各种文学作品、写作技巧和文学理论。请以文雅的语言与用户讨论文学相关话题。如果用户说退出、再见等词汇，请礼貌地结束对话。"
            },
            "3": {
                "name": "生活顾问",
                "prompt": "你是一个生活顾问，可以提供关于健康、情感、职业发展等方面的建议。请以温暖、体贴的语调回答问题。如果用户说退出、再见等词汇，请礼貌地结束对话。"
            }
        }

        # 存储不同会话的聊天历史
        self.store = {}

        # 当前选择的主题
        self.current_topic = None
        self.chain = None
        self.with_msg_hist = None

    def get_session_history(self, session_id):
        """根据session_id获取聊天历史"""
        if session_id not in self.store:
            self.store[session_id] = InMemoryChatMessageHistory()
        return self.store[session_id]

    def setup_chain(self, topic_prompt):
        """根据主题设置对话链"""
        # 创建带占位符的prompt模板
        prompt = ChatPromptTemplate.from_messages([
            ("system", topic_prompt),
            MessagesPlaceholder(variable_name="messages"),
        ])

        parser = StrOutputParser()

        # 构建chain
        self.chain = prompt | self.model | parser

        # 注入聊天历史
        self.with_msg_hist = RunnableWithMessageHistory(
            self.chain,
            get_session_history=self.get_session_history,
            input_messages_key="messages"
        )

    def display_topics(self):
        """显示可选主题"""
        print("欢迎使用特定主题聊天系统")
        print("请选择您想讨论的主题：")
        for key, topic in self.topics.items():
            print(f"{key}. {topic['name']}")
        print("0. 退出系统")

    def select_topic(self):
        """选择聊天主题"""
        while True:
            self.display_topics()
            choice = input("请输入主题编号: ").strip()
            if choice in self.topics:
                self.current_topic = self.topics[choice]
                self.setup_chain(self.current_topic["prompt"])
                print(f"已选择主题：{self.current_topic['name']}")
                print("现在可以开始对话了，输入'换主题'可重新选择主题")
                return True
            else:
                print("无效选择，请重新输入")

    def chat(self):
        """开始聊天"""
        if not self.with_msg_hist:
            print("请先选择主题")
            return
        session_id = f"topic_{list(self.topics.keys())[list(self.topics.values()).index(self.current_topic)]}"
        while True:
            user_input = input(f"{self.current_topic['name']} - 您: ").strip()

            if user_input in ['换主题', '更换主题', 'change']:
                return True

            # 调用LangChain进行对话
            response = self.with_msg_hist.invoke(
                {"messages": [HumanMessage(content=user_input)]},
                config={'configurable': {'session_id': session_id}}
            )
            print(f"AI助手: {response}")


    def run(self):
        """运行聊天系统"""
        while True:
            if self.select_topic():
                should_continue = self.chat()
                if not should_continue:
                    break

if __name__ == "__main__":
    # 创建并运行聊天系统
    chat_system = TopicChatSystem()
    chat_system.run()