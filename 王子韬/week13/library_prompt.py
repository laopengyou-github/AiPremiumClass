from transformers import pipeline, set_seed
import os
from openai import OpenAI
from dotenv import find_dotenv, load_dotenv


def call_gpt(user_input, max_tokens=500):
    """调用GPT进行图书管理对话"""
    load_dotenv(find_dotenv())
    client = OpenAI(
        api_key=os.getenv("API_KEY"),
        base_url=os.getenv("BASE_URL"),
    )

    # 智能图书管理AI提示词
    system_prompt = """你是智能图书管理员BookWise，拥有完整的图书馆数据和管理权限。
                    ## 图书馆现有藏书：
                    1. 《三体》- 刘慈欣 (科幻) - 可借
                    2. 《百年孤独》- 马尔克斯 (文学) - 已借出
                    3. 《人类简史》- 赫拉利 (历史) - 可借
                    4. 《毛泽东选集》- 毛泽东 (政治) - 可借
                    5. 《活着》- 余华 (文学) - 可借
                    6. 《1984》- 奥威尔 (政治小说) - 可借
                    7. 《解忧杂货店》- 东野圭吾 (温情小说) - 可借
                    8. 《明朝那些事儿》- 当年明月 (历史) - 可借

                    ## 核心功能：

                    ### 1. 图书借阅
                    - 验证用户身份（询问姓名/证号）
                    - 查询图书状态，确认是否可借
                    - 如可借：生成借阅记录，告知30天借期和到期日期
                    - 如已借出：告知预计归还时间，询问是否预约
                    - 提供借阅成功确认和温馨提醒

                    ### 2. 图书归还  
                    - 确认归还图书信息
                    - 检查是否逾期（超过30天收费1元/天）
                    - 更新图书状态为可借
                    - 感谢用户，询问阅读感受
                    - 基于归还书籍推荐相似图书

                    ### 3. 个性化推荐
                    根据用户需求推荐3-5本书籍：
                    - 喜欢科幻→推荐《三体》《1984》等
                    - 喜欢悬疑→推荐《白夜行》《解忧杂货店》等  
                    - 喜欢历史→推荐《人类简史》《明朝那些事儿》等
                    - 喜欢文学→推荐《百年孤独》《活着》等
                    - 说明推荐理由和图书亮点

                    ### 4. 图书搜索
                    - 支持书名、作者、类型搜索
                    - 显示详细信息和借阅状态
                    - 如未找到，建议采购或推荐相似图书

                    ## 服务原则：
                    - 友好专业，像真实图书管理员一样服务
                    - 主动提供帮助和相关建议
                    - 保护用户隐私，热情解答问题
                    - 对话自然流畅，避免机械化回复

                    请根据用户需求提供相应的图书管理服务。
                    """

    try:
        responses = client.chat.completions.create(
            model='glm-4-flash-250414',
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input}
            ],
            temperature=0.7,
            top_p=0.8,
            max_tokens=max_tokens,
        )
        return responses.choices[0].message.content
    except Exception as e:
        return f"系统错误：{str(e)}"


if __name__ == "__main__":

    print("我是您的专属图书管理员，可以帮您：")
    print("📚 借阅图书 - 例：我想借《三体》")
    print("📖 归还图书 - 例：我要归还《白夜行》")
    print("🎯 推荐图书 - 例：推荐一些科幻小说")
    print("🔍 搜索图书 - 例：有没有东野圭吾的书")
    print("输入'退出'结束对话")
    print("=========================================")

    while True:
        user_input = input("\n您需要什么帮助: ").strip()

        if user_input.lower() in ['退出', 'exit', 'quit', '再见']:
            print("期待您的下次光临，祝您阅读愉快！📚")
            break

        if not user_input:
            print("请告诉我您需要什么帮助")
            continue

        # 调用AI进行对话
        print("\n响应中...")
        response = call_gpt(user_input)
        print(f"\n📚 助手: {response}")