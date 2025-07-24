# agent & GPT
## agent
##### 1、加载环境 load_dotenv()
##### 2、通过from langchain_openai import ChatOpenAI来创建 llm;
##### 3、创建提示词模板，使用from langchain import hub来加载模板 prompt = hub.pull("hwchase17/openai-functions-agent");
##### 4、构建工具 search=TavilySearchResults(max_results=2); 只检索2个结果
##### 5、工具列表 tools = [search];  可以支持多个工具tools = [search,retriever_tool]，agent来调用这里的工具
##### 5、构建agent agent = create_tool_calling_agent(llm=model, prompt=prompt, tools=tools);
##### 6、创建agent executor，就是调用 executor = AgentExecutor(agent=agent, tools=tools, verbose=True);
##### 7、运行agent msgs = executor.invoke({"input":"北京明天天气如何"});
### Tips:如果在工具列表添加其他工具如下：
##### 7.1 tools = [search,retriever_tool]，使用
create_retriever_tool: 这是一个函数，用于基于给定的检索器创建一个工具。
retriever: 在这里传递的检索器是上面创建的 retriever。
"book_retriever": 这是给工具指定的名称，用于标识和引用这个工具。
description: 提供工具的描述信息，在构建智能助手或复杂应用时，这个描述信息非常有用。它可以帮助用户或其他开发者了解这个工具的功能和用途。
## GPT
##### 具体操作需要看代码，nano2主要是添加了self-attention，注意self-attention添加的位置
