# graphrag可以通过ollama本地部署，但使用国产模型未能在线部署
## 基础RAG分为：构建索引1~3、检索和生成4~5
### 1、Load：首先使用document_loaders来加载数据(还有web连接或者pdf的加载函数)
### 2、split：文本数据太大了，先拆分为小块(chunks)
### 3、store：需要用某种方式来存储和索引我们拆分出来的小块chunks，方便我们检索，一般用vectorstore，embedding模型
### 4、retrieve：根据用户输入，使用retrieve来检索相关拆分的东西 as_retrieve
### 5、genetate：使用流 的方式进行对问题进行传递，生成答案
