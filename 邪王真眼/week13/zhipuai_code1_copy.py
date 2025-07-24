import os
from dotenv import load_dotenv, find_dotenv

# # 查找 .env 文件的路径
# path = find_dotenv()
# # 加载 .env 文件, 条目写入系统环境变量
# load_dotenv(path)

# # 通过os包读取系统环境变量
# print(os.environ['api_key'])
# print(os.environ['base_url'])

load_dotenv(find_dotenv())
