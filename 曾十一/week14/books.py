import sqlite3
import os

DB_PATH = "/mnt/data_1/zfy/homework/books.db"

# 如果已存在旧数据库，可以删除（可选）
if os.path.exists(DB_PATH):
    os.remove(DB_PATH)

# 连接数据库（自动创建）
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

# 创建 books 表
cursor.execute('''
CREATE TABLE IF NOT EXISTS books (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    author TEXT NOT NULL,
    category TEXT,
    summary TEXT,
    stock INTEGER DEFAULT 1,
    available BOOLEAN DEFAULT 1
)
''')

# 插入示例图书数据
# 插入更多示例图书数据（共15本）
books = [
    ("三体", "刘慈欣", "科幻", "讲述人类与外星文明首次接触及文明冲突的史诗级小说", 5, 1),
    ("百年孤独", "加西亚·马尔克斯", "文学", "魔幻现实主义杰作，描绘布恩迪亚家族七代人的传奇。", 3, 1),
    ("追风筝的人", "卡勒德·胡赛尼", "小说", "关于友情与救赎的故事，情感深刻感人至深。", 4, 1),
    ("人工智能简史", "李开复", "科技", "通俗易懂地讲述人工智能的发展历史与未来。", 6, 1),
    ("小王子", "圣埃克苏佩里", "童话", "充满哲思的童话故事，适合儿童与成人共同阅读。", 10, 1),
    ("活着", "余华", "文学", "通过普通人命运展现中国现代历史变迁，深沉而感人。", 5, 1),
    ("乌合之众", "古斯塔夫·勒庞", "心理学", "揭示群体心理与集体行为背后的规律，对社会学有深远影响。", 7, 1),
    ("人类简史", "尤瓦尔·赫拉利", "历史", "从认知革命到科技革命，深入剖析人类文明发展。", 6, 1),
    ("沉默的大多数", "王小波", "杂文", "犀利幽默，充满思想力的随笔集，反映中国社会文化。", 4, 1),
    ("时间简史", "斯蒂芬·霍金", "科普", "通俗解读宇宙起源、黑洞与时间的本质。", 3, 1),
    ("红楼梦", "曹雪芹", "古典文学", "中国古代四大名著之一，展现封建贵族家庭的盛衰与人情冷暖。", 5, 1),
    ("自控力", "凯利·麦格尼格尔", "心理学", "心理学畅销书，帮助读者掌握自律技巧与行为管理。", 8, 1),
    ("解忧杂货店", "东野圭吾", "小说", "以信件形式讲述治愈人心的故事，温暖动人。", 6, 1),
    ("影响力", "罗伯特·西奥迪尼", "心理学", "研究人类行为背后的说服力与社会影响机制。", 5, 1),
    ("Python编程：从入门到实践", "埃里克·马瑟斯", "编程", "零基础学 Python 的实用指南，项目实战丰富。", 10, 1)
]


cursor.executemany('''
INSERT INTO books (title, author, category, summary, stock, available)
VALUES (?, ?, ?, ?, ?, ?)
''', books)

conn.commit()
conn.close()

print("✅ 数据库 books.db 已成功创建，含示例图书数据！")
