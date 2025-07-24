#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
图书管理系统聊天机器人
基于LangChain实现的智能图书管理助手
"""

import getpass
import os
import time
import sys
import json
import random
from datetime import datetime
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain


class LibraryChatbot:
    """图书管理系统聊天机器人"""
    
    def __init__(self):
        """初始化聊天机器人"""
        # 设置API密钥
        if not os.environ.get("ZHIPU_API_KEY"):
            os.environ["ZHIPU_API_KEY"] = getpass.getpass("输入ZHIPU_API_KEY: ")
        
        # 初始化GLM模型
        self.model = ChatOpenAI(
            model="glm-4-flash-250414",
            base_url="https://open.bigmodel.cn/api/paas/v4",
            api_key=os.environ["ZHIPU_API_KEY"]
        )
        
        # 定义系统提示词
        self.system_prompt = """
你是一个专业的图书管理助手。你的职责包括：

1. 回答关于图书借阅、查询的各种问题
2. 根据用户需求推荐合适的书籍
3. 管理用户的借阅记录
4. 提供图书相关信息和服务

请用友好、专业的语言回答用户的问题。
"""
        
        # 创建对话链和记忆
        self.memory = ConversationBufferMemory(return_messages=True)
        self.conversation = ConversationChain(
            llm=self.model,
            memory=self.memory,
            verbose=False
        )
        
        # 借阅记录存储
        self.borrowing_records = {}
    
    def recommend_books(self, user_description=""):
        """推荐书籍"""
        if not user_description or user_description.strip() == "":
            # 无描述时推荐经典名著
            recommendation_prompt = """
请推荐3部经典名著，要求：
1. 每本书包含：书名、作者、出版时间、内容简介
2. 选择不同国家和时代的经典作品
3. 适合不同年龄段的读者
4. 格式要清晰易读

请用中文回答，格式如下：
1. **书名**
   作者：xxx
   出版时间：xxx
   简介：xxx

2. **书名**
   作者：xxx
   出版时间：xxx
   简介：xxx

3. **书名**
   作者：xxx
   出版时间：xxx
   简介：xxx
"""
        else:
            # 根据用户描述推荐书籍
            recommendation_prompt = f"""
根据用户描述「{user_description}」，推荐3本最相关的书籍，要求：
1. 每本书包含：书名、作者、出版时间、内容简介
2. 书籍要与用户需求高度相关
3. 选择不同难度和角度的书籍
4. 格式要清晰易读

请用中文回答，格式如下：
1. **书名**
   作者：xxx
   出版时间：xxx
   简介：xxx

2. **书名**
   作者：xxx
   出版时间：xxx
   简介：xxx

3. **书名**
   作者：xxx
   出版时间：xxx
   简介：xxx
"""
        
        try:
            messages = [
                SystemMessage(content="你是一个专业的图书推荐助手，请根据用户需求推荐最合适的书籍。"),
                HumanMessage(content=recommendation_prompt)
            ]
            
            ai_response = self.model.invoke(messages)
            response = f"📚 根据您的描述「{user_description}」，为您推荐以下书籍：\n\n" if user_description else "📚 为您推荐以下经典名著：\n\n"
            response += ai_response.content
            
            return response
            
        except Exception as e:
            # 如果AI推荐失败，返回错误信息
            return f"❌ 推荐过程中出现错误: {str(e)}"
    
    def borrow_book(self, user_id, book_title):
        """借阅书籍"""
        if user_id not in self.borrowing_records:
            self.borrowing_records[user_id] = []
        
        # 记录借阅
        borrow_record = {
            "book_title": book_title,
            "borrow_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "return_date": None
        }
        
        self.borrowing_records[user_id].append(borrow_record)
        
        return f"✅ 借阅成功！您已成功借阅「{book_title}」。\n借阅时间：{borrow_record['borrow_date']}"
    
    def return_book(self, user_id, book_title):
        """归还书籍"""
        if user_id not in self.borrowing_records:
            return "❌ 您没有借阅记录。"
        
        for record in self.borrowing_records[user_id]:
            if book_title in record['book_title'] or record['book_title'] in book_title:
                if record['return_date'] is None:
                    record['return_date'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    return f"✅ 归还成功！您已归还「{record['book_title']}」。\n归还时间：{record['return_date']}"
                else:
                    return f"❌ 该书籍「{record['book_title']}」已经归还过了。"
        
        return f"❌ 未找到您借阅的书籍「{book_title}」。"
    
    def check_borrowing_records(self, user_id):
        """查询借阅记录"""
        if user_id not in self.borrowing_records or not self.borrowing_records[user_id]:
            return "📋 您目前没有借阅记录。"
        
        response = f"📋 您的借阅记录：\n\n"
        for i, record in enumerate(self.borrowing_records[user_id], 1):
            response += f"{i}. **{record['book_title']}**\n"
            response += f"   借阅时间：{record['borrow_date']}\n"
            if record['return_date']:
                response += f"   归还时间：{record['return_date']}\n"
                response += f"   状态：已归还 ✅\n"
            else:
                response += f"   状态：借阅中 📖\n"
            response += "\n"
        
        return response
    
    def process_command(self, user_input, user_id="default_user"):
        """处理特殊命令"""
        user_input_lower = user_input.lower().strip()
        
        # 查询借阅记录命令 - 优先级最高
        if user_input_lower in ["借阅记录", "我的借阅", "借书记录", "记录"]:
            return self.check_borrowing_records(user_id)
        
        # 推荐书籍命令
        if user_input_lower.startswith("推荐") or "推荐" in user_input_lower:
            description = user_input.replace("推荐", "").strip()
            return self.recommend_books(description)
        
        # 归还书籍命令
        elif user_input_lower.startswith("归还") or "还" in user_input_lower:
            book_title = user_input.replace("归还", "").replace("还", "").strip()
            if book_title:
                return self.return_book(user_id, book_title)
            else:
                return "❌ 请指定要归还的书籍名称。"
        
        # 借阅书籍命令 - 优先级最低，避免与"借阅记录"冲突
        elif user_input_lower.startswith("借阅") or ("借" in user_input_lower and "记录" not in user_input_lower):
            book_title = user_input.replace("借阅", "").replace("借", "").strip()
            if book_title:
                return self.borrow_book(user_id, book_title)
            else:
                return "❌ 请指定要借阅的书籍名称。"
        
        return None  # 不是特殊命令，交给AI处理
    
    def stream_chat(self, user_input, user_id="default_user"):
        """与图书管理助手进行流式对话"""
        try:
            # 检查是否是特殊命令
            command_response = self.process_command(user_input, user_id)
            if command_response:
                print(f"\n🤖 {command_response}")
                return command_response
            
            # 构建消息列表
            messages = [SystemMessage(content=self.system_prompt)]
            messages.append(HumanMessage(content=user_input))
            
            # 流式获取AI响应
            print("\n🤖 AI: ", end="", flush=True)
            full_response = ""
            
            for chunk in self.model.stream(messages):
                if hasattr(chunk, 'content') and chunk.content:
                    content = chunk.content
                    print(content, end="", flush=True)
                    full_response += content
            
            print()  # 换行
            
            return full_response
            
        except Exception as e:
            error_msg = f"对话过程中出现错误: {str(e)}"
            print(f"\n❌ {error_msg}")
            return error_msg
    
    def start_chat(self):
        """开始聊天"""
        print("📚 欢迎使用图书管理系统聊天机器人！")
        print("💡 我可以帮助您进行图书咨询、推荐和借阅管理")
        print("📝 输入 'quit' 或 'exit' 退出聊天")
        print("\n🔧 特殊功能命令：")
        print("   • 推荐 [描述] - 根据描述推荐书籍（无描述则推荐名著）")
        print("   • 借阅 [书名] - 借阅指定书籍")
        print("   • 归还 [书名] - 归还指定书籍")
        print("   • 借阅记录 - 查看您的借阅记录")
        print("-" * 50)
        
        while True:
            try:
                user_input = input("\n👤 你: ").strip()
                
                if user_input.lower() in ['quit', 'exit', '退出']:
                    print("\n👋 感谢使用图书管理系统，再见！")
                    break
                    
                elif not user_input:
                    print("⚠️  请输入有效的问题")
                    continue
                
                # 使用流式输出
                self.stream_chat(user_input)
                
            except KeyboardInterrupt:
                print("\n\n👋 聊天已中断，再见！")
                break
            except Exception as e:
                print(f"\n❌ 发生错误: {str(e)}")


def main():
    """主函数"""
    chatbot = LibraryChatbot()
    chatbot.start_chat()


if __name__ == "__main__":
    main() 