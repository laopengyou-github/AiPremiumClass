#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NLP主题聊天系统
基于LangChain实现的多轮对话NLP专家助手
"""

import getpass
import os
import time
import sys
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain.chat_models import init_chat_model
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain


class NLPChatbot:
    """NLP主题聊天机器人"""
    
    def __init__(self):
        """初始化聊天机器人"""
        # 设置API密钥
        if not os.environ.get("GOOGLE_API_KEY"):
            os.environ["GOOGLE_API_KEY"] = getpass.getpass("输入API_KEY: ")
        
        # 初始化模型
        self.model = init_chat_model("gemini-2.0-flash", model_provider="google_genai")
        
        # 定义系统提示词
        self.system_prompt = """
你是一个专业的NLP（自然语言处理）专家助手。你的职责包括：

1. 回答关于自然语言处理的各种问题
2. 解释NLP相关的概念、算法和技术
3. 提供代码示例和实现建议
4. 讨论最新的NLP研究进展和模型
5. 帮助解决NLP项目中的技术问题

请用清晰、专业但易懂的语言回答用户的问题。
如果用户的问题超出NLP范围，请礼貌地引导回NLP主题。
"""
        
        # 创建对话链和记忆
        self.memory = ConversationBufferMemory(return_messages=True)
        self.conversation = ConversationChain(
            llm=self.model,
            memory=self.memory,
            verbose=False
        )
        
        # 聊天历史
        self.chat_history = []
    
    def add_to_history(self, role, content):
        """添加消息到聊天历史"""
        self.chat_history.append({
            "role": role, 
            "content": content, 
            "timestamp": time.strftime("%H:%M:%S")
        })
    
    def display_history(self):
        """显示聊天历史"""
        print("\n" + "="*50)
        print("聊天历史:")
        print("="*50)
        for msg in self.chat_history:
            role_icon = "👤" if msg["role"] == "user" else "🤖"
            print(f"{msg['timestamp']} {role_icon} {msg['role'].upper()}: {msg['content']}")
        print("="*50 + "\n")
    
    def stream_chat(self, user_input):
        """与NLP专家进行流式对话"""
        try:
            # 添加用户消息到历史
            self.add_to_history("user", user_input)
            
            # 构建完整的消息列表
            messages = [SystemMessage(content=self.system_prompt)]
            
            # 添加历史对话
            for msg in self.chat_history[:-1]:  # 除了当前用户输入
                if msg["role"] == "user":
                    messages.append(HumanMessage(content=msg["content"]))
                else:
                    messages.append(AIMessage(content=msg["content"]))
            
            # 添加当前用户输入
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
            
            # 添加AI响应到历史
            self.add_to_history("assistant", full_response)
            
            return full_response
            
        except Exception as e:
            error_msg = f"对话过程中出现错误: {str(e)}"
            self.add_to_history("system", error_msg)
            print(f"\n❌ {error_msg}")
            return error_msg
    
    def clear_history(self):
        """清空聊天历史"""
        self.chat_history.clear()
        self.memory.clear()
        print("✅ 聊天历史已清空")
    
    def start_chat(self):
        """开始聊天"""
        print("🤖 欢迎使用NLP专家聊天系统！")
        print("💡 我可以帮助你解答NLP相关的问题")
        print("📝 输入 'quit' 或 'exit' 退出聊天")
        print("📋 输入 'history' 查看聊天历史")
        print("🔄 输入 'clear' 清空聊天历史")
        print("-" * 50)
        
        while True:
            try:
                user_input = input("\n👤 你: ").strip()
                
                if user_input.lower() in ['quit', 'exit', '退出']:
                    print("\n👋 感谢使用NLP专家聊天系统，再见！")
                    break
                    
                elif user_input.lower() == 'history':
                    self.display_history()
                    continue
                    
                elif user_input.lower() == 'clear':
                    self.clear_history()
                    continue
                    
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
    chatbot = NLPChatbot()
    chatbot.start_chat()


if __name__ == "__main__":
    main() 