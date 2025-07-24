from dotenv import find_dotenv, load_dotenv
import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


class LibraryManagementSystem:
    def __init__(self):
        load_dotenv(find_dotenv())

        self.model = ChatOpenAI(
            model="glm-4-flash-250414",
            base_url=os.environ['BASE_URL'],
            api_key=os.environ['API_KEY'],
            temperature=0.7
        )

        self.store = {}
        self.setup_chain()

    def get_session_history(self, session_id):
        if session_id not in self.store:
            self.store[session_id] = InMemoryChatMessageHistory()
        return self.store[session_id]

    def setup_chain(self):
        system_prompt = """
        你是智能图书管理员BookWise，拥有完整的图书馆数据和管理权限。
            ## 图书馆现有藏书及简介：
            1. 《三体》- 刘慈欣 (科幻) - 可借
               简介：描述了地球文明与三体文明的信息交流、生死搏杀及两个文明在宇宙中的兴衰历程。硬科幻巨作，探讨文明冲突与人性选择。
            
            2. 《百年孤独》- 马尔克斯 (文学) - 已借出
               简介：魔幻现实主义经典，讲述布恩迪亚家族七代人的传奇故事，反映拉丁美洲百年历史的孤独与轮回。
            
            3. 《人类简史》- 赫拉利 (历史) - 可借
               简介：从认知革命、农业革命到科学革命，全景式讲述人类发展史，思考人类未来走向。
            
            4. 《毛泽东选集》- 毛泽东 (政治) - 可借
               简介：收录毛泽东重要著作和文章，体现马克思主义中国化的理论成果和实践经验。
            
            5. 《活着》- 余华 (文学) - 可借
               简介：讲述福贵的人生悲剧，在苦难中展现生命的韧性和活着本身的意义，催人深思。
            
            6. 《1984》- 奥威尔 (政治小说) - 可借
               简介：反乌托邦经典，描绘极权社会的恐怖图景，探讨自由、真理与人性的意义。
            
            7. 《解忧杂货店》- 东野圭吾 (温情小说) - 可借
               简介：神奇杂货店连接过去与未来，通过温暖的故事传递希望，治愈人心的暖心之作。
            
            8. 《明朝那些事儿》- 当年明月 (历史) - 可借
               简介：以幽默风趣的语言讲述明朝276年历史，让历史变得生动有趣，深受读者喜爱。
            
            ## 核心功能：
            1. 图书借阅：验证用户身份，查询图书状态，生成借阅记录（30天借期）
            2. 图书归还：确认归还信息，检查逾期情况，更新图书状态
            3. 图书咨询：基于图书简介为读者提供详细的内容介绍和阅读价值分析
            4. 个性化推荐：根据用户兴趣和图书特点提供精准推荐
            5. 图书搜索：支持书名、作者、类型搜索，显示详细信息和借阅状态
            
            ## 服务原则：
            - 根据图书内容特点进行精准推荐
            - 如果用户说退出、再见等词汇，请结束对话
            
        请根据用户需求提供相应的图书管理服务。
        """

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="messages"),
        ])

        chain = prompt | self.model | StrOutputParser()

        self.with_msg_hist = RunnableWithMessageHistory(
            chain,
            get_session_history=self.get_session_history,
            input_messages_key="messages"
        )

    def run(self):
        print("智能图书管理系统")
        print("我可以帮您：借阅图书、归还图书、图书推荐、图书搜索、图书咨询")

        session_id = "library_session"

        while True:
            user_input = input("您需要什么帮助: ").strip()

            if not user_input:
                continue

            try:
                response = self.with_msg_hist.invoke(
                    {"messages": [HumanMessage(content=user_input)]},
                    config={'configurable': {'session_id': session_id}}
                )
                print(f"BookWise: {response}")

                # 检查AI是否礼貌地结束了对话
                if any(word in response.lower() for word in ['再见', '结束', '退出', '拜拜', 'goodbye']):
                    break

            except Exception as e:
                print(f"系统错误：{str(e)}")


if __name__ == "__main__":
    library_system = LibraryManagementSystem()
    library_system.run()