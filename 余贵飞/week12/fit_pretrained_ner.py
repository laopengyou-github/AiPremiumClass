# 将文件写入kaggle的输出文件夹中
# %%wirtefile doushabao_ner_ddp.py

# 导入相应的依赖
from transformers import AutoModelForTokenClassification, AutoTokenizer,DataCollatorForTokenClassification
from transformers import TrainingArguments, Trainer
import torch
import evaluate  # pip install evaluate
import seqeval   # pip install seqeval
from datasets import load_dataset
import numpy as np
import torch.distributed as dist
import torch.multiprocessing as mp
import os

# 设置分布式环境
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

# 清理分布式环境
def cleanup():
    dist.destroy_process_group()

def train(rank, world_size):
    # 初始化分布式环境
    setup(rank, world_size)
    # 加载数据集
    orignal_data = load_dataset("doushabao4766/msra_ner_k_V3")

    #构造实体列表数据用于过滤不包含实体的数据
    # class_label_list = [1, 2, 3, 4, 5, 6]

    # 过滤掉源数据中token 长度大于512的数据
    filter_data = orignal_data.filter(lambda item: len(item['tokens']) <=512)

    # 手动标记实体类别标签
    tags = ['0','B-PER','I-PER','B-ORG','I-ORG','B-LOC','I-LOC']

    # 模型标签转换字典
    id2lbl = {i:tag for i, tag in enumerate(tags)}
    lbl2id = {tag:i for i, tag in enumerate(tags)}

    # 初始化模型使用 模型名称为：google-bert/bert-base-chinese 最终预测类别为7个对应最后输出层神经元个数
    model = AutoModelForTokenClassification.from_pretrained('google-bert/bert-base-chinese', 
                                                            num_labels=7,
                                                            id2label=id2lbl,
                                                            label2id=lbl2id)

    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-chinese")

    # 数据进行处理将数据处理为能够数据模型的数据
    def data_input_proc(item):
        # 输入文本转换模型输入token索引
        input_data = tokenizer(item['tokens'], truncation=True, add_special_tokens = False, padding = True, max_length = 512, is_split_into_words = True)
        # 数据是已经对齐的只需要将 item 中的 ner_tags 添加到 input data 中(不需要)
        input_data['labels'] = item['ner_tags']
        return input_data

    # 处理过滤结束的数据，组装模型训练输入数据
    ds_filter = filter_data.map(data_input_proc, batched = True)

    # 将数据集中需要输入模型的字段转换为张量
    ds_filter.set_format('torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'labels'])


    # metric 方法
    def compute_metric(result):
        # result 是一个tuple (predicts, labels)
        
        # 获取评估对象
        seqeval = evaluate.load('seqeval')
        predicts, labels = result
        predicts = np.argmax(predicts, axis=2)
        
        # 准备评估数据
        predicts = [[tags[p] for p,l in zip(ps,ls) if l != -100]
                    for ps,ls in zip(predicts,labels)]
        labels = [[tags[l] for p,l in zip(ps,ls) if l != -100]
                    for ps,ls in zip(predicts,labels)]
        results = seqeval.compute(predictions=predicts, references=labels)

        return results


    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer, padding=True)

    # 记录GPU的编号
    local_rank = rank
    # 将模型移动到GPU上
    model.to(local_rank)

    # TrainingArguments
    args = TrainingArguments(
        output_dir="ner_train",  # 模型训练工作目录（tensorboard，临时模型存盘文件，日志）
        num_train_epochs = 5,    # 训练 epoch
        save_safetensors=False,  # 设置False保存文件可以通过torch.load加载
        per_device_train_batch_size=16,  # 训练批次
        per_device_eval_batch_size=16,
        report_to='tensorboard',  # 训练输出记录
        eval_strategy="epoch",
        local_rank=local_rank, # 当前进程的RANK
        fp16=True, # 使用混合精度训练
        lr_scheduler_type='linear', # 动态学习率
        warmup_steps=600, # 预热步数
        ddp_find_unused_parameters=False # 优化DDP性能
    )

    # 使用过滤处理的数据进行训练
    trainer = Trainer(
        model,
        args,
        train_dataset=ds_filter['train'],
        eval_dataset=ds_filter['test'],
        data_collator=data_collator,
        compute_metrics=compute_metric
    )
    trainer.train()

def main():
    world_size = torch.cuda.device_count()
    mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)

if __name__ == '__main__':
    try:
        main()
    finally:
    # 清理分布式进程组
        cleanup()
   