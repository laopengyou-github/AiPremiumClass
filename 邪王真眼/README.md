本项目用来记录NLP课程作业，每个文件夹是一个小任务，previous开头的是往期作业，week开头的是当期作业，其余为小工具或框架。
以_copy结尾的文件是拷贝来的，不是本人编写。
本项目设计了支持插件的trainer类，并模仿成熟框架以config的形式传递配置信息，其中插件可以在训练关键节点（类似训练开始前，每个batch结束后等）执行操作，在解决固定化流程的问题上可以方便的增删功能，但由于后期课程任务形式变化的较频繁，所以并非所有小任务都是用这个trainer。
关于数据集，部分较大且能从网上下载的不在git仓库中，其余的均已上传。 TODO：开一个不含数据集的分支


| 周数 | 课程名称 | 课程内容 | 作业内容 |
|------|----------|----------|----------|
| 第一周 | 基本介绍 | 自然语言处理行业简介（目标价值、职业需求分析） | 无 |
| 第二周 | 深度学习基本原理 | 向量运算，矩阵运算，求导法则。反向传播原理，学习率与优化器，dropout，DNN、CNN、RNN。 | 实现一个基于交叉熵的多分类任务，任务可以自拟。 |
| 第三周 | 深度学习处理文本 | 激活函数、损失函数、池化层等神经网络组件应用 | 自己设计文本任务目标，使用rnn进行多分类。 |
| 第四周 | 中文分词相关 | 正向/负向/双向最大切分等分词方法。使用神经网络训练分词模型。tfidf原理及计算方式，基于tfidf文本检索，摘要抽取。基于信息熵和聚合度的新词发现方法。 | 实现文本的全切分 |
| 第五周 | 词向量 | 基于窗口的训练，基于语言模型的训练，基于共现矩阵的训练。Huffman树和负采样的训练提速方法。词相似度计算，句向量，文本相似度计算，文本聚类。 | 实现基于kmeans的类内距离计算，筛选优质类别。 |
| 第六周 | 预训练模型 | ngram语言模型的实现。神经网络语言模型的实现。基于语言模型的文本分类，文本纠错，可读性增强等任务。Bert、GPT等语言模型的训练，transformer结构的思想及实现。 | 计算bert中的可训练参数数量。 |
| 第七周 | 文本分类问题 | 基于支持向量机，朴素贝叶斯，lstm，gru，cnn，rcnn，bert等方式的文本分类实现。标签不平衡，标注数据稀疏等问题的处理思路。 | 在电商评论数据集上做文本分类试验。 |
| 第八周 | 文本匹配 | 编辑距离，jaccard距离，bm25，词向量等方式实现文本相似度计算。交互式和匹配式文本匹配的区别和应用场景。基于三元组损失的训练方式。 | 修改表示形文本匹配代码，使用三元组损失函数训练。 |
| 第九周 | 序列标注 | NER，分词等。CRF应用，维特比解码，beam search等。基于词表和正则表达式完成关键信息的抽取。 | 用bert实现ner。 |
| 第十周 | 生成式任务 | 基于rnn、transformer的实现。soft attention，hard attention，local attention，self attention等注意力机制的作用。 | 使用bert结构完成自回归语言模型训练。 |
| 第十一周 | 大语言模型第一讲 | sft，instruction following, in context learning, hallucination等相关知识 | 基于上周的的新闻标题和内容数据实现sft训练。 |
| 第十二周 | 大语言模型第二讲 | prompt engineering，AI agent 相关知识。llama,chatglm,baichuan,qwen等模型结构，Rope，Alibi位置编码等相关知识 |  阅读代码整理开源模型结构差异，填在Excel中。 |
| 第十三周 | 大语言模型微调 | reward model，rlhf，lora，p-tuning等相关知识。 | 使用lora做ner任务，数据可以用之前的。 |
| 第十四周 | 大语言模型RAG | RAG(Retrieval-Augmented Generation),langchain,向量数据库等相关知识。deepspeed原理，流水线并行，数据并行，混合精度等相关概念。基于bpe的词表构建等知识。GPT-4V，Flamingo,LLava，sora等多模态模型相关知识 。 | 尝试实现bpe构建词表，并完成文本编解码。 |
| 第十五周 | 知识图谱 | 实体抽取，关系抽取，属性抽取的实现。封闭式和开放式的知识抽取。图数据库建表语句和查询语句的使用。基于匹配和模型生成的nl2sql实现。 | 尝试安装使用neo4j数据库完成知识图谱问答。 |
| 第十六周 | 对话系统 | 聊天型，任务型，问答型对话机器人的需求及实现差异。工业界基于场景脚本的任务型对话机器人实现。 | 为对话系统加入重听能力 |
| 第十七周 | 推荐系统 | 协同过滤，内容召回，热点召回等。排序学习（pointwise、pairwise、listwise）。离散特征embedding，连续特征的分箱处理，特征组合。google play store, youtube等落地推荐系统方案 | 实现itemCF |


| 周数 | 课程名称 | 作业内容 |
|------|----------|----------|
| 第一周 | pytorch基础 | 1.练习课题代码示例（尝试使用不同的参数及数据，观察结果）<br>2.总结归纳课题知识点（推荐markdown格式文档或电子笔记）<br>3.预习高等数学微分求导过程，线性代数矩阵一般性概念及内积运算 |
| 第二周 | pytorch逻辑回归 | 1、使用sklearn数据集训练逻辑回归模型；<br>2、调整学习率，样本数据拆分比率，观察训练结果；<br>3、训练后模型参数保存到文件，在另一个代码中加载参数实现预测功能；<br>4、总结逻辑回归运算及训练相关知识点 |
| 第三周 | pytorch神经网络 | 1. 使用pytorch搭建神经网络模型，实现对KMNIST数据集的训练。<https://pytorch.org/vision/stable/generated/torchvision.datasets.KMNIST.html#torchvision.datasets.KMNIST><br>2. 尝试调整模型结构（变更神经元数量，增加隐藏层）来提升模型预测的准确率<br>3. 调试超参数，观察学习率和批次大小对训练的影响 |
| 第四周 | pytorch模型训练相关要素 | 1. 搭建的神经网络，使用olivettiface数据集进行训练；<br>2. 结合归一化和正则化来优化网络模型结构，观察对比loss结果；<br>3. 尝试不同optimizer对模型进行训练，观察对比loss结果；<br>4. 注册kaggle并尝试激活Accelerator，使用GPU加速模型训练 |
| 第五周 | 语言模型及词向量相关知识 | 1. 实现基于豆瓣top250图书评论的简单推荐系统（TF-IDF及BM25两种算法实现）；<br>2. 使用自定义的文档文本，通过fasttext训练word2vec训练词向量模型，并计算词汇间的相关度。（选做：尝试tensorboard绘制词向量可视化图）；<br>3. 使用课堂示例cooking.stackexchange.txt，使用fasttext训练文本分类模型。（选做：尝试使用Kaggle中的Fake News数据集训练文本分类模型）<https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification> |
| 第六周 | 循环神经网络 | 1. 实验使用不同的RNN结构，实现一个人脸图像分类器。至少对比2种以上结构训练损失和准确率差异，如：LSTM、GRU、RNN、BiRNN等。要求使用tensorboard，提交代码及run目录和可视化截图。<https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_olivetti_faces.html><br>2. 使用RNN实现一个天气预测模型，能预测1天和连续5天的最高气温。要求使用tensorboard，提交代码及run目录和可视化截图。数据集：<https://www.kaggle.com/datasets/smid80/weatherww2> |
| 第七周 | 基于深度学习的RNN文本分类 | 1. 使用豆瓣电影评论数据完成文本分类处理：文本预处理，加载、构建词典。（评论得分1～2表示positive取值：1，评论得分4～5代表negative取值：0）<https://www.kaggle.com/datasets/utmhikari/doubanmovieshortcomments><br>2. 加载处理后文本构建词典、定义模型、训练、评估、测试；<br>3. 尝试不同分词工具进行文本分词，观察模型训练结果 |
| 第八周 | attention与transformer | 1. 使用中文对联数据集训练带有attention的seq2seq模型，利用tensorboard跟踪。<https://www.kaggle.com/datasets/jiaminggogogo/chinese-couplets><br>2. 尝试encoder hidden state不同的返回形式（concat和add）；<br>3. 编写并实现seq2seq attention版的推理实现 |
| 第九周 | transformer | 1. 根据课堂演示代码，完成自定义transformer模型架构的搭建及训练。保存模型；<br>2. 尝试加载模型进行推理（选做部分）；<br>3. 总结Transformer模型架构文档 |
| 第十周 | Bert结构预训练模型及资源 | 1. 根据提供的kaggle JD评论语料进行文本分类训练<https://www.kaggle.com/datasets/dosonleung/jd_comment_with_label><br>2. 调整模型训练参数，添加tensorboard跟踪，对比bert冻结和不冻结之间的训练差异；<br>3. 保存模型进行分类预测 |
| 第十一周 | 命名实体识别及预训练模型微调 | 1. 参考课堂案例，使用指定的数据集，编写代码实现ner模型训练和推流。<https://huggingface.co/datasets/doushabao4766/msra_ner_k_V3><br>2. 完成预测结果的实体抽取。输入："双方确定了今后发展中美关系的指导方针。" 输出：[{"entity":"ORG","content":"中"},{"entity":"ORG","content":"美"}]<br>3. 整理Dataset、Trainer、TrainingArgument、DataCollator、Evaluate 知识点，总结文档 |
| 第十二周 | 预训练模型微调技巧 | 1. 利用上周NER模型训练任务代码，复现课堂案例中：动态学习率、混合精度、DDP训练实现；<br>2. 利用课堂案例，实现分布式DDP模型训练。存盘后加载实现推理 |
| 第十三周 | LLM模型prompt开发及大模型应用 | 1. 安装ollama，下载模型并用代码方式调用；<br>2. 利用OpenAI API 调用远端大模型API，调试参数观察输出结果的差异；<br>3. 利用大模型提示词设计一个智能图书管理AI；<br>功能:实现图书借阅和归还。根据喜好为读者推荐图书。 |
| 第十四周 | LangChain及RAG原理 | 1. 通过langchain实现特定主题聊天系统，支持多轮对话。<br>2. 借助langchain实现图书管理系统开发扩展，通过图书简介为借阅读者提供咨询。 |
| 第十五周 | RAG相关技术及Agent应用 | 1. 根据课堂RAG示例，完成外部文档导入并进行RAG检索的过程。<br>外部PDF文档：https://storage.googleapis.com/deepmind-media/Era-of-Experience%20/The%20Era%20of%20Experience%20Paper.pdf<br># 使用 langchain_community.document_loaders.PDFMinerLoader 加载 PDF 文件。<br>docs = PDFMinerLoader(path).load()<br>2. 使用graphrag构建一篇小说（自主选择文档）的RAG知识图，实现本地和全局问题的问答。（截图代码运行结果） |
| 第十六周 | GPT系列模型搭建训练及优化 | 1. 使用langchain_agent实现pdf和TAVILY工具组合动态调用的实现。<br>2. 使用nano-gpt2.0模型，训练一份中文语料并测试生成。 |
