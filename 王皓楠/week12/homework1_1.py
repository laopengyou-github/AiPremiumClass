import os
import numpy as np
import torch
import evaluate
import argparse
from datasets import load_dataset
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer
)


def prepare_labels():
    """生成NER任务的标签列表及映射关系（id2label/label2id）"""
    # MSRA NER标准实体类型：PER(人物)、LOC(地点)、ORG(机构)
    entities = ['PER', 'LOC', 'ORG']
    # 构建标签列表：O(非实体) + B-XXX(实体开始) + I-XXX(实体内部)
    tags = ['O']
    for entity in entities:
        tags.append(f'B-{entity}')
        tags.append(f'I-{entity}')
    # 生成映射字典
    id2label = {i: tag for i, tag in enumerate(tags)}
    label2id = {tag: i for i, tag in enumerate(tags)}
    return tags, id2label, label2id


def preprocess_function(examples, tokenizer, max_length=256):
    """
    数据预处理函数：将文本转换为模型输入格式
    Args:
        examples: 数据集样本（批量）
        tokenizer: 预训练模型的tokenizer
        max_length: 最大序列长度
    Returns:
        处理后的模型输入（input_ids, attention_mask, labels）
    """
    # MSRA数据集的文本已按字符拆分，直接使用is_split_into_words=True
    texts = examples['tokens']  # 格式：[[char1, char2, ...], [char1, ...]]
    encoding = tokenizer(
        texts,
        truncation=True,
        max_length=max_length,
        is_split_into_words=True,  # 告知tokenizer输入已按词（字符）拆分
        padding='max_length',
        add_special_tokens=False  # 不添加[CLS]/[SEP]，保持与标签长度一致
    )

    # 处理标签（截断到max_length，与输入序列长度匹配）
    labels = []
    for lbl in examples['ner_tags']:
        if len(lbl) > max_length:
            labels.append(lbl[:max_length])
        else:
            # 不足max_length的部分补0（实际训练中会被mask掉，不影响）
            labels.append(lbl + [0] * (max_length - len(lbl)))
    encoding['labels'] = labels
    return encoding


def compute_metrics(eval_preds, tags):
    """
    评估指标计算函数：使用seqeval计算NER的precision/recall/f1
    Args:
        eval_preds: 评估预测结果（tuple: (logits, labels)）
        tags: 标签列表（用于将id转换为标签文本）
    Returns:
        评估指标字典（含overall和各实体类型的precision/recall/f1）
    """
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=2)  # 从logits中取预测概率最大的标签

    # 过滤掉被mask的标签（labels=-100的位置）
    true_predictions = [
        [tags[p] for p, l in zip(pred_row, label_row) if l != -100]
        for pred_row, label_row in zip(predictions, labels)
    ]
    true_labels = [
        [tags[l] for p, l in zip(pred_row, label_row) if l != -100]
        for pred_row, label_row in zip(predictions, labels)
    ]

    # 使用seqeval计算指标
    seqeval_metric = evaluate.load('seqeval')
    results = seqeval_metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "overall_precision": results["overall_precision"],
        "overall_recall": results["overall_recall"],
        "overall_f1": results["overall_f1"],
        "overall_accuracy": results["overall_accuracy"],
    }


def train(local_rank, tags, id2label, label2id):
    """训练函数：加载数据、初始化模型、配置训练参数并启动训练"""
    # 1. 加载数据集（MSRA NER数据集）
    print(f"[Rank {local_rank}] 加载数据集...")
    dataset = load_dataset('msra_ner')  # 修正数据集名称（原'ds_msra_ner'可能为自定义名称）
    print(f"数据集结构: {dataset}")

    # 2. 加载tokenizer
    model_name = "bert-base-chinese"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # 3. 预处理数据集
    print(f"[Rank {local_rank}] 预处理数据...")
    tokenized_dataset = dataset.map(
        lambda x: preprocess_function(x, tokenizer),
        batched=True,  # 批量处理（加速）
        remove_columns=dataset["train"].column_names  # 移除不需要的原始列
    )
    # 转换为PyTorch张量格式
    tokenized_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    # 4. 初始化模型
    print(f"[Rank {local_rank}] 加载模型...")
    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=len(tags),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True  # 忽略预训练头与新任务头的尺寸不匹配
    )

    # 5. 配置训练参数
    training_args = TrainingArguments(
        output_dir="./ner_train_results",  # 训练输出目录
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        warmup_steps=100,  # 学习率预热步数
        learning_rate=2e-5,  # BERT常用学习率
        lr_scheduler_type="linear",
        evaluation_strategy="epoch",  # 每轮epoch评估一次
        save_strategy="epoch",  # 每轮epoch保存一次模型
        logging_dir="./logs",  # 日志目录（供tensorboard查看）
        logging_steps=100,
        fp16=torch.cuda.is_available(),  # 若有GPU支持，启用混合精度训练
        load_best_model_at_end=True,  # 训练结束后加载最佳模型
        metric_for_best_model="overall_f1",  # 以f1作为最佳模型指标
        local_rank=local_rank,  # 分布式训练Rank
        ddp_find_unused_parameters=False,  # 优化DDP性能
        save_safetensors=True  # 使用安全的safetensors格式保存模型
    )

    # 6. 初始化Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        tokenizer=tokenizer,
        data_collator=DataCollatorForTokenClassification(tokenizer=tokenizer),
        compute_metrics=lambda x: compute_metrics(x, tags)  # 绑定评估函数
    )

    # 7. 启动训练
    print(f"[Rank {local_rank}] 开始训练...")
    trainer.train()
    print(f"[Rank {local_rank}] 训练完成!")


def main():
    # 解析命令行参数（分布式训练需要local_rank）
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=-1, help="分布式训练的本地Rank")
    args = parser.parse_args()

    # 准备标签映射
    tags, id2label, label2id = prepare_labels()
    print(f"NER标签列表: {tags}")

    # 启动训练
    train(args.local_rank, tags, id2label, label2id)


if __name__ == "__main__":
    main()