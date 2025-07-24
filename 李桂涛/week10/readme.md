## huggingface使用Bert做训练任务
### 一、Bert
##### 1、有三个embedding，1、token转换的，2、词属于哪个句子，3、位置编码；
##### 2、常见的预训练任务：MLM(masked language model)用15%的词中80%用mask的token表示10%用原始token表示，10%用随机token表示；NSP(next sentence prediction):使用上一句预测下一句
### 二、huggingface
##### 1、下载huggingface模型可以使用官方的下载工具huggingface-cli，pip install "huggingface_hub[hf_transfer]";
##### 2、Windows在power shall中设置环境变量：$env:HF_ENDPOINT = "https://alpha.hf-mirror.com/"；mac || linux：export HF_ENDPOINT = https://alpha.hf-mirror.com/
##### 3、下载模型命令：huggingface-cli download microsoft/bitnet-b1.58-2B-4T --local-dir bitnet-1.58，也就是huggingface-cli download <model_id> --local-dir <local dir>
### 三、魔搭
##### 1、国内可以用魔搭：pip install modelscope 
##### 2、下载模型命令：modelscope download --model='Qwen/Qwen2.5-0.5B-Inst'
### Tips
##### 其余都是使用集成包transformers。 
