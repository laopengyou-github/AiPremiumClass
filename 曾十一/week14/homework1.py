from dotenv import find_dotenv, load_dotenv
import os
from langchain_openai import ChatOpenAI  # 调用 OpenAI/GLM 模型
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage  # 多轮对话消息结构
from langchain_core.output_parsers import StrOutputParser  # 输出解析为字符串

from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import FileChatMessageHistory  # 聊天历史持久化工具

# ✅ 设置统一的历史保存目录
HISTORY_DIR = "/mnt/data_1/zfy/homework/history"
os.makedirs(HISTORY_DIR, exist_ok=True)  # 如果目录不存在则自动创建

# ✅ 定义历史记录读取/创建函数（按 session_id 存储 JSON 文件）
def get_session_hist(session_id: str):
    history_file = os.path.join(HISTORY_DIR, f"history_{session_id}.json")
    return FileChatMessageHistory(history_file)

if __name__ == '__main__':
    # ✅ 加载环境变量（如 API 密钥和 base_url）
    load_dotenv(find_dotenv())

    # ✅ 初始化语言模型
    model = ChatOpenAI(
        model="glm-4-flash-250414",
        base_url=os.environ['base_url'],
        api_key=os.environ['api_key'],
        temperature=0.7
    )

    # ✅ 定义 Prompt 模板（系统消息 + 占位历史）
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个得力的助手。尽你所能使用{lang}回答所有问题。"),
        MessagesPlaceholder(variable_name="messages"),
    ])

    # ✅ 构建 Chain：prompt → model → 解析器
    parser = StrOutputParser()
    chain = prompt | model | parser

    # ✅ 构建可记忆对话链（自动注入历史记录）
    with_msg_hist = RunnableWithMessageHistory(
        chain,
        get_session_history=get_session_hist,
        input_messages_key="messages"
    )

    # ✅ 获取当前会话 ID（区分不同聊天历史）
    session_id = input("请输入会话 session_id（用于区别多个用户）：").strip()

    # ✅ 主聊天循环
    while True:
        user_input = input('用户输入的Message（输入 quit 退出）：')
        if user_input.lower() == "quit":
            break

        # 构造输入并调用链
        response = with_msg_hist.invoke(
            {
                "messages": [HumanMessage(content=user_input)],
                "lang": "中文"  # 你可以根据需要动态设定
            },
            config={"configurable": {"session_id": session_id}}
        )

        # 输出模型返回的结果
        print("AI Message:", response)
