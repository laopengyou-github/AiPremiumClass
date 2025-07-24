

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1'

import numpy as np
from transformers import AutoModelForTokenClassification, AutoTokenizer,DataCollatorForTokenClassification
from transformers import TrainingArguments, Trainer
import torch
import evaluate  # pip install evaluate
import seqeval   # pip install seqeval
from datasets import load_dataset
import torch.distributed as dist
import torch.multiprocessing as mp

# 设置分布式环境
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

# 清理分布式环境
def cleanup():
    dist.destroy_process_group()
def extract_entities(sentence, model, tokenizer, id2label):
    tokens = list(sentence)

    # 分词 + 放入模型设备
    inputs = tokenizer(tokens, return_tensors="pt", is_split_into_words=True, truncation=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # 模型预测
    outputs = model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=-1).squeeze().tolist()

    # 标签解码
    labels = [id2label[idx] for idx in predictions]

    # BIO 解码成实体块
    entities = []
    entity = None
    for token, label in zip(tokens, labels):
        if label.startswith("B-"):
            if entity:
                entities.append(entity)
            entity = {"entity": label[2:], "content": token}
        elif label.startswith("I-") and entity and label[2:] == entity["entity"]:
            entity["content"] += token
        else:
            if entity:
                entities.append(entity)
                entity = None
    if entity:
        entities.append(entity)

    return entities

def train(rank, world_size):
    setup(rank, world_size)
    # 数据集
    dataset = load_dataset('doushabao4766/msra_ner_k_V3')
    label_list = dataset["train"].features["ner_tags"].feature.names
    label2id = {label: i for i, label in enumerate(label_list)}
    id2label = {i: label for label, i in label2id.items()}

    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)
        labels = []
        for i, label in enumerate(examples["ner_tags"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:
                    label_ids.append(label[word_idx])
                else:
                    label_ids.append(label[word_idx] if label[word_idx] != -100 else -100)
                previous_word_idx = word_idx
            labels.append(label_ids)
        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    tokenized_dataset = dataset.map(tokenize_and_align_labels, batched=True)
    model = AutoModelForTokenClassification.from_pretrained(
        'google-bert/bert-base-chinese', 
        num_labels=len(label_list),
        id2label=id2label,
        label2id=label2id
    )
    model.to(local_rank)

    args = TrainingArguments(
        output_dir="./ner-model",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        # learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        report_to='tensorboard',  # 训练输出记录
        num_train_epochs=3,
        # weight_decay=0.01,
        # save_total_limit=2,
        save_safetensors=False,  # 设置False保存文件可以通过torch.load加载
        local_rank=local_rank,   # 当前进程 RANK
        fp16=True,               # 使用混合精度
        lr_scheduler_type='linear',  # 动态学习率
        warmup_steps=100,        # 预热步数
        ddp_find_unused_parameters=False  # 优化DDP性能
    )

    def compute_metrics(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)
        true_labels = [[id2label[l] for l in label if l != -100] for label in labels]
        true_predictions = [
            [id2label[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        return {
            "f1": seqeval.metrics.f1_score(true_labels, true_predictions),
            "precision": seqeval.metrics.precision_score(true_labels, true_predictions),
            "recall": seqeval.metrics.recall_score(true_labels, true_predictions),
        }

    data_collator = DataCollatorForTokenClassification(tokenizer)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        # tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        # callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )
    trainer.train()
    cleanup()


        

def main():
    world_size = torch.cuda.device_count()
    mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)

if __name__ == "__main__":
    main()
    

