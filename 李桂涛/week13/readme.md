# LLM模型prompt开发
### ollama
##### 1、直接官网下载就可以，然后根据电脑配置运行不同模型进行本地聊天，终端形式：查看ollama本地模型：ollama list,ollama 运行：ollama run deepseek-r1:8b，ollama会在本地显示localhost:11434，分为chat和generate
##### 2、openaiaip调用需要安装openai python-dotenv，来加载api key形式调用模型，role就是角色，system就是系统指定的‘你是XXX(一个开发工程师)’，user：用户，assistance：提示，user可以有多个来提示，这些作为历史消息给api调用的大模型client
