from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer, TrainingArguments
from datasets import load_dataset
import torch
import numpy as np

# 加载数据集
dataset = load_dataset("doushabao4766/msra_ner_k_V3")

# 标签映射
label_list = dataset["train"].features["ner_tags"].feature.names
id2label = {i: label for i, label in enumerate(label_list)}
label2id = {label: i for i, label in enumerate(label_list)}

# 加载预训练模型和分词器
model_name = "bert-base-chinese"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(
    model_name,
    num_labels=len(label_list),
    id2label=id2label,
    label2id=label2id
)

# 数据预处理
def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True,
        padding="max_length",
        max_length=128
    )
    
    labels = []
    for i, tags in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(tags[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)
    
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

# 处理数据集
tokenized_datasets = dataset.map(
    tokenize_and_align_labels,
    batched=True,
    remove_columns=dataset["train"].column_names
)

# 训练参数
training_args = TrainingArguments(
    output_dir="./ner_results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    save_strategy="no",
    logging_dir="./logs",
)

# 初始化Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
)

# 训练模型
trainer.train()

# 保存模型
model.save_pretrained("./ner_model")
tokenizer.save_pretrained("./ner_model")

# 预测函数
def predict_entities(text):
    inputs = tokenizer(
        list(text),  # 将文本拆分为字符列表
        is_split_into_words=True,
        return_tensors="pt",
        truncation=True,
        padding=True
    )
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    predictions = torch.argmax(outputs.logits, dim=-1).squeeze().tolist()
    
    # 获取有效token的预测结果
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    word_ids = inputs.word_ids(batch_index=0)
    
    entities = []
    current_entity = None
    
    for idx, (token, word_id) in enumerate(zip(tokens, word_ids)):
        if word_id is None:  # 跳过特殊token
            continue
            
        label = id2label[predictions[idx]]
        
        if label != "O":
            entity_type = label.split("-")[-1]  # 提取实体类型
            char_position = word_id
            
            # 检查是否为实体起始或独立实体
            if label.startswith("B-") or not current_entity or current_entity["type"] != entity_type:
                if current_entity:  # 保存前一个实体
                    entities.append(current_entity)
                current_entity = {
                    "entity": entity_type,
                    "content": text[char_position],
                    "start": char_position
                }
            else:
                # 合并连续实体
                current_entity["content"] += text[char_position]
        elif current_entity:
            entities.append(current_entity)
            current_entity = None
    
    if current_entity:
        entities.append(current_entity)
    
    # 转换为要求的输出格式
    result = []
    for ent in entities:
        # 如果实体长度>1，拆分为单个字符
        if len(ent["content"]) > 1:
            for char in ent["content"]:
                result.append({"entity": ent["entity"], "content": char})
        else:
            result.append(ent)
    
    return result

# 测试预测
input_text = "双方确定了今后发展中美关系的指导方针。"
output = predict_entities(input_text)
print(output)