import os
from dotenv import load_dotenv, find_dotenv
from flask import Flask, request, jsonify
from openai import OpenAI
import json

# 加载环境变量
load_dotenv(find_dotenv())

# 创建OpenAI客户端
client = OpenAI(
    api_key=os.environ.get('API_KEY'),
    base_url=os.environ.get('BASE_URL', None)
)

# 模拟图书馆数据库
library_db = {
    "books": [
        {"id": "B10001", "title": "人类简史", "author": "尤瓦尔·赫拉利", "category": "历史", "available": True},
        {"id": "B10002", "title": "未来简史", "author": "尤瓦尔·赫拉利", "category": "历史", "available": True},
        {"id": "B10003", "title": "三体", "author": "刘慈欣", "category": "科幻", "available": True},
        {"id": "B10004", "title": "球状闪电", "author": "刘慈欣", "category": "科幻", "available": False},
        {"id": "B10005", "title": "沙丘", "author": "弗兰克·赫伯特", "category": "科幻", "available": True},
        {"id": "B10006", "title": "银河帝国", "author": "艾萨克·阿西莫夫", "category": "科幻", "available": True},
        {"id": "B10007", "title": "解忧杂货店", "author": "东野圭吾", "category": "文学", "available": True},
        {"id": "B10008", "title": "白夜行", "author": "东野圭吾", "category": "文学", "available": False}
    ],
    "readers": {},
    "categories": {
        "科幻": ["三体", "球状闪电", "沙丘", "银河帝国"],
        "历史": ["人类简史", "未来简史"],
        "文学": ["解忧杂货店", "白夜行"]
    }
}

app = Flask(__name__)

def generate_response(messages):
    """调用OpenAI API生成回复"""
    try:
        response = client.chat.completions.create(
            model="glm-4-flash",  # 使用可用的模型
            messages=messages,
            temperature=0.7,
            max_tokens=150
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"API调用错误: {e}")
        return "抱歉，我暂时无法回答这个问题。"

def process_library_action(user_message, reader_id="default"):
    """处理图书馆相关操作"""
    # 初始化读者数据
    if reader_id not in library_db["readers"]:
        library_db["readers"][reader_id] = {"borrowed_books": [], "preferences": []}
    
    # 检查借阅请求
    if "借阅" in user_message or "借" in user_message:
        for book in library_db["books"]:
            if book["title"] in user_message:
                if book["available"]:
                    # 处理借阅
                    library_db["books"][library_db["books"].index(book)]["available"] = False
                    library_db["readers"][reader_id]["borrowed_books"].append(book["id"])
                    return f"已成功为您借阅《{book['title']}》，借阅编号：{book['id']}。请在30天内归还。"
                else:
                    return f"抱歉，《{book['title']}》目前已被借出，无法借阅。"
        return "抱歉，未找到您想借阅的图书。"
    
    # 检查归还请求
    if "归还" in user_message or "还" in user_message:
        for book_id in library_db["readers"][reader_id]["borrowed_books"]:
            for book in library_db["books"]:
                if book["id"] == book_id and book["title"] in user_message:
                    # 处理归还
                    library_db["books"][library_db["books"].index(book)]["available"] = True
                    library_db["readers"][reader_id]["borrowed_books"].remove(book_id)
                    return f"已成功归还《{book['title']}》。感谢您的借阅！"
        return "抱歉，未找到您需要归还的图书记录。"
    
    # 检查推荐请求
    if "推荐" in user_message or "喜欢" in user_message or "类似" in user_message:
        # 提取用户偏好
        preferences = []
        for category in library_db["categories"]:
            for book_title in library_db["categories"][category]:
                if book_title in user_message:
                    preferences.append(category)
                    break
        
        # 如果检测到新偏好，更新用户偏好
        if preferences and preferences[0] not in library_db["readers"][reader_id]["preferences"]:
            library_db["readers"][reader_id]["preferences"].extend(preferences)
        
        # 根据偏好推荐图书
        if preferences:
            recommended_books = []
            for category in preferences:
                for book_title in library_db["categories"][category]:
                    for book in library_db["books"]:
                        if book["title"] == book_title and book["available"]:
                            recommended_books.append(book)
                            break
            
            if recommended_books:
                response = "根据您的喜好，为您推荐以下图书：\n"
                for i, book in enumerate(recommended_books[:3], 1):
                    response += f"{i}. 《{book['title']}》 - {book['author']} ({book['category']})\n"
                return response + "需要我帮您查询这些书的馆藏情况吗？"
        
        return "请您提供更多信息，例如您喜欢的作者或书籍类型，以便我为您推荐合适的图书。"
    
    # 如果没有匹配的图书馆操作，转由AI处理
    return None

@app.route('/library/chat', methods=['POST'])
def library_chat():
    """处理图书馆聊天请求"""
#     request={
#     "reader_id": "user123",
#     "message": "我想借阅《三体》"
# }
    data = request.json
    user_message = data.get('message', '')
    reader_id = data.get('reader_id', 'default')
    
    # 检查是否是图书馆相关操作
    library_response = process_library_action(user_message, reader_id)
    if library_response:
        return jsonify({"response": library_response})
    
    # 如果不是图书馆操作，调用AI生成回复
    # 构建对话历史
    messages = [
        {"role": "system", "content": "你是一个智能图书馆助手，具备以下能力：\n- 管理图书借阅与归还流程\n- 根据读者的阅读喜好推荐相关图书\n- 提供图书信息查询服务（如作者、分类、简介）\n- 保持专业、礼貌的沟通态度"},
        {"role": "user", "content": user_message}
    ]
    
    # 调用AI生成回复
    ai_response = generate_response(messages)
    return jsonify({"response": ai_response})

if __name__ == '__main__':
    app.run(debug=True)