import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments,
    pipeline
)
import torch
from torch.utils.data import Dataset

# ========================== 1.数据准备 ==========================
class CommentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,                          #需要处理的文本tokens
            max_length=self.max_length,    #最大长度
            padding="max_length",          #不为none时，用max_length填充
            truncation=True,               #不为none时，超过max_length裁剪
            return_tensors="pt"            #返回pytorch张量,numpy为np，torch为pt，none时返回python类型对象
        )
        
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(label, dtype=torch.long)
        }

# 加载数据
try:
    df = pd.read_excel("C:/Users/ligt/.cache/kagglehub/datasets/dosonleung/jd_comment_with_label/versions/1/jd_comment_data.xlsx")
    texts = df["评价内容(content)"].tolist()
    labels = df["评分（总分5分）(score)"].tolist()
except Exception as e:
    print(f"数据加载失败: {str(e)}")
    exit()

# 检查标签是否需要编码（如果标签是字符串）
if isinstance(labels[0], str):
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    labels = le.fit_transform(labels)

# 划分数据集
train_texts, val_texts, train_labels, val_labels = train_test_split(
    texts, labels, test_size=0.2, random_state=42
)

# ========================== 2.模型配置 ==========================
tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
num_labels = len(set(labels))

def create_model(freeze_bert=False):
    """创建模型并选择性冻结BERT参数"""
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-chinese",
        num_labels=num_labels
    )
    
    if freeze_bert:
        # 冻结BERT参数，仅训练分类层
        for param in model.bert.parameters():
            param.requires_grad = False
    else:
        # 解冻所有参数
        for param in model.parameters():
            param.requires_grad = True
            
    return model

# ========================== 3.训练设置 ==========================
def compute_metrics(p):
    """自定义评估指标"""
    preds = np.argmax(p.predictions, axis=1)
    return {"accuracy": accuracy_score(p.label_ids, preds)}

# 创建数据集
train_dataset = CommentDataset(train_texts, train_labels, tokenizer)
val_dataset = CommentDataset(val_texts, val_labels, tokenizer)

def run_experiment(freeze_bert=False, lr=5e-5, epochs=3):
    """运行训练实验"""
    model = create_model(freeze_bert)
    
    training_args = TrainingArguments(
        output_dir=f"./results_{'frozen' if freeze_bert else 'unfrozen'}",
        num_train_epochs=epochs,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        evaluation_strategy="epoch",
        logging_dir=f"./logs_{'frozen' if freeze_bert else 'unfrozen'}",
        logging_steps=50,
        save_strategy="epoch",
        learning_rate=lr,
        report_to="tensorboard",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )

    print(f"\n{'='*30} 开始训练（BERT冻结：{freeze_bert}） {'='*30}")
    trainer.train()
    
    return model

# ========================== 4.执行训练 ==========================
# 实验1：冻结BERT参数
frozen_model = run_experiment(freeze_bert=True, lr=5e-5, epochs=3)

# 实验2：不冻结BERT参数
unfrozen_model = run_experiment(freeze_bert=False, lr=3e-5, epochs=5)

# ========================== 5.模型保存 ==========================
best_model_dir = "./best_model"
frozen_model.save_pretrained(best_model_dir)
tokenizer.save_pretrained(best_model_dir)

# ========================== 6.预测示例 ==========================
classifier = pipeline(
    "text-classification",
    model=best_model_dir,
    tokenizer=best_model_dir,
    device=0 if torch.cuda.is_available() else -1
)

# 测试预测
sample_text = "这个商品质量非常好，物流速度也很快！"
result = classifier(sample_text)
print(f"\n预测结果示例：\n输入文本：{sample_text}\n分类结果：{result}")