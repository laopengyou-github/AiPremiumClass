import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup
)
from datasets import load_dataset
from tqdm import tqdm
import numpy as np


class NERTrainer:
    """NER模型训练器，支持动态学习率、混合精度和DDP训练"""

    def __init__(self, model_name='google-bert/bert-base-chinese'):
        self.model_name = model_name
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # 实体类别定义
        self.entities = ['O'] + list({'movie', 'name', 'game', 'address', 'position',
                                      'company', 'scene', 'book', 'organization', 'government'})
        self.tags = ['O']
        for entity in self.entities[1:]:
            self.tags.append('B-' + entity.upper())
            self.tags.append('I-' + entity.upper())

        self.entity_index = {entity: i for i, entity in enumerate(self.entities)}
        self.id2label = {i: tag for i, tag in enumerate(self.tags)}
        self.label2id = {tag: i for i, tag in enumerate(self.tags)}

        # 初始化tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def prepare_data(self):
        """准备训练数据"""
        print("加载数据集...")
        ds = load_dataset('nlhappy/CLUE-NER')

        # 处理实体标签
        def entity_tags_proc(item):
            text_len = len(item['text'])
            tags = [0] * text_len  # 初始值为'O'

            entities = item['ents']
            for ent in entities:
                indices = ent['indices']
                label = ent['label']
                tags[indices[0]] = self.entity_index[label] * 2 - 1  # B-标签
                for idx in indices[1:]:
                    tags[idx] = self.entity_index[label] * 2  # I-标签
            return {'ent_tag': tags}

        ds1 = ds.map(entity_tags_proc)

        # 处理输入数据
        def data_input_proc(item):
            batch_texts = [list(text) for text in item['text']]
            input_data = self.tokenizer(
                batch_texts,
                truncation=True,
                add_special_tokens=False,
                max_length=512,
                is_split_into_words=True,
                padding='max_length'
            )
            # 对标签进行padding
            input_data['labels'] = [tag + [0] * (512 - len(tag)) for tag in item['ent_tag']]
            return input_data

        ds2 = ds1.map(data_input_proc, batched=True)

        # 设置数据格式为PyTorch张量
        ds2.set_format('torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'labels'])

        return ds2

    def setup_model_and_optimizer(self, learning_rate=1e-5):
        """设置模型和优化器"""
        print("初始化模型...")
        self.model = AutoModelForTokenClassification.from_pretrained(
            self.model_name,
            num_labels=len(self.tags),
            id2label=self.id2label,
            label2id=self.label2id
        )
        self.model.to(self.device)

        # 参数分组：BERT层和分类层使用不同的学习率
        param_optimizer = list(self.model.named_parameters())
        bert_params, classifier_params = [], []

        for name, params in param_optimizer:
            if 'bert' in name:
                bert_params.append(params)
            else:
                classifier_params.append(params)

        # 设置不同的学习率和权重衰减
        param_groups = [
            {'params': bert_params, 'lr': learning_rate},
            {'params': classifier_params, 'weight_decay': 0.1, 'lr': learning_rate * 100}
        ]

        self.optimizer = optim.AdamW(param_groups)

    def train_with_dynamic_lr(self, train_dataloader, epochs=5):
        """使用动态学习率进行训练"""
        print("\n开始训练（动态学习率）...")

        # 设置学习率调度器
        total_steps = len(train_dataloader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=100,  # 预热步数
            num_training_steps=total_steps
        )

        for epoch in range(epochs):
            self.model.train()
            pbar = tqdm(train_dataloader, desc=f'Epoch {epoch + 1}/{epochs}')

            for batch in pbar:
                # 将数据移到GPU
                batch = {k: v.to(self.device) for k, v in batch.items()}

                self.optimizer.zero_grad()
                outputs = self.model(**batch)
                loss = outputs.loss

                loss.backward()
                self.optimizer.step()
                scheduler.step()  # 更新学习率

                # 更新进度条显示
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'bert_lr': f'{scheduler.get_lr()[0]:.2e}',
                    'cls_lr': f'{scheduler.get_lr()[1]:.2e}'
                })

    def train_with_mixed_precision(self, train_dataloader, epochs=5):
        """使用混合精度训练"""
        print("\n开始训练（混合精度）...")

        # 设置学习率调度器
        total_steps = len(train_dataloader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=100,
            num_training_steps=total_steps
        )

        # 初始化梯度缩放器
        scaler = torch.cuda.amp.GradScaler()

        for epoch in range(epochs):
            self.model.train()
            pbar = tqdm(train_dataloader, desc=f'Epoch {epoch + 1}/{epochs}')

            for batch in pbar:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                self.optimizer.zero_grad()

                # 使用自动混合精度
                with torch.cuda.amp.autocast():
                    outputs = self.model(**batch)
                    loss = outputs.loss

                # 缩放损失并反向传播
                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()
                scheduler.step()

                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'bert_lr': f'{scheduler.get_lr()[0]:.2e}',
                    'cls_lr': f'{scheduler.get_lr()[1]:.2e}'
                })

    def save_model(self, save_path='./ner_model'):
        """保存模型"""
        print(f"\n保存模型到 {save_path}")
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)

        # 保存标签映射
        import json
        with open(f'{save_path}/label_mappings.json', 'w', encoding='utf-8') as f:
            json.dump({
                'id2label': self.id2label,
                'label2id': self.label2id,
                'tags': self.tags
            }, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    # 初始化训练器
    trainer = NERTrainer()

    # 准备数据
    dataset = trainer.prepare_data()
    train_dataloader = DataLoader(
        dataset['train'],
        shuffle=True,
        batch_size=16
    )

    # 方式1：动态学习率训练
    print("\n=== 方式1：动态学习率训练 ===")
    trainer.setup_model_and_optimizer(learning_rate=1e-5)
    trainer.train_with_dynamic_lr(train_dataloader, epochs=2)

    # 方式2：混合精度训练
    print("\n=== 方式2：混合精度训练 ===")
    trainer.setup_model_and_optimizer(learning_rate=1e-5)
    trainer.train_with_mixed_precision(train_dataloader, epochs=2)

    # 保存模型
    trainer.save_model('./ner_model_advanced')

    print("\n训练完成！")