import os
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Config
from transformers import BertTokenizerFast
from torch.utils.data import Dataset, DataLoader
import jieba
from tqdm import tqdm
import wandb

class ChineseTextDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512):
        """初始化数据集
        Args:
            data_path (str): 训练数据路径
            tokenizer: 分词器
            max_length (int): 最大序列长度
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # 读取数据
        with open(data_path, 'r', encoding='utf-8') as f:
            self.text = f.read()
            
        # 分词
        self.tokens = list(jieba.cut(self.text))
        self.token_ids = tokenizer.convert_tokens_to_ids(self.tokens)
        
    def __len__(self):
        # 返回token_ids的长度减去max_length的值
        return len(self.token_ids) - self.max_length
        
    def __getitem__(self, idx):
        # 获取指定索引的token_ids
        chunk = self.token_ids[idx:idx + self.max_length]
        # 将chunk的前max_length-1个元素转换为tensor，作为输入
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        # 将chunk的后max_length-1个元素转换为tensor，作为标签
        y = torch.tensor(chunk[1:], dtype=torch.long)
        # 返回输入和标签
        return x, y

class ChineseGPT2Trainer:
    def __init__(self, model_name='gpt2', train_data_path=None):
        """初始化训练器
        Args:
            model_name (str): 模型名称
            train_data_path (str): 训练数据路径
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 初始化tokenizer和model
        self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-chinese')
        config = GPT2Config(
            vocab_size=len(self.tokenizer),
            n_positions=512,
            n_ctx=512,
            n_embd=768,
            n_layer=6,
            n_head=12
        )
        self.model = GPT2LMHeadModel(config)
        self.model.to(self.device)
        
        if train_data_path:
            self.dataset = ChineseTextDataset(
                train_data_path,
                self.tokenizer
            )
            
    def train(self, 
              batch_size=8,
              epochs=10,
              learning_rate=3e-4,
              warmup_steps=1000,
              gradient_accumulation_steps=1):
        """训练模型
        Args:
            batch_size (int): 批次大小
            epochs (int): 训练轮数
            learning_rate (float): 学习率
            warmup_steps (int): 预热步数
            gradient_accumulation_steps (int): 梯度累积步数
        """
        # 初始化wandb
        wandb.init(project="chinese-gpt2", config={
            "batch_size": batch_size,
            "epochs": epochs,
            "learning_rate": learning_rate,
            "warmup_steps": warmup_steps
        })
        
        dataloader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4
        )
        
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate
        )
        
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.1,
            total_iters=warmup_steps
        )
        
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            progress_bar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{epochs}')
            
            for step, (x, y) in enumerate(progress_bar):
                x = x.to(self.device)
                y = y.to(self.device)
                
                outputs = self.model(x, labels=y)
                loss = outputs.loss / gradient_accumulation_steps
                loss.backward()
                
                if (step + 1) % gradient_accumulation_steps == 0:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                
                total_loss += loss.item()
                progress_bar.set_postfix({'loss': loss.item()})
                
                wandb.log({
                    "loss": loss.item(),
                    "learning_rate": scheduler.get_last_lr()[0]
                })
            
            avg_loss = total_loss / len(dataloader)
            print(f'Epoch {epoch+1}, Average Loss: {avg_loss:.4f}')
            
            # 保存模型
            if (epoch + 1) % 5 == 0:
                self.save_model(f'model_epoch_{epoch+1}')
        
        wandb.finish()
    
    def generate(self, prompt, max_length=100, temperature=0.7, top_p=0.9):
        """生成文本
        Args:
            prompt (str): 提示文本
            max_length (int): 最大生成长度
            temperature (float): 温度参数
            top_p (float): top-p采样参数
        Returns:
            str: 生成的文本
        """
        self.model.eval()
        with torch.no_grad():
            # 对提示文本进行编码
            inputs = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
            
            # 生成
            outputs = self.model.generate(
                inputs,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                no_repeat_ngram_size=3,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
            
            # 解码
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return generated_text
    
    def save_model(self, path):
        """保存模型
        Args:
            path (str): 保存路径
        """
        os.makedirs(path, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
    
    def load_model(self, path):
        """加载模型
        Args:
            path (str): 模型路径
        """
        self.model = GPT2LMHeadModel.from_pretrained(path)
        self.tokenizer = BertTokenizerFast.from_pretrained(path)
        self.model.to(self.device)

def main():
    # 使用示例
    trainer = ChineseGPT2Trainer(
        train_data_path='data/train.txt'  # 替换为实际的训练数据路径
    )
    
    # 训练
    trainer.train(
        batch_size=8,
        epochs=10
    )
    
    # 生成示例
    prompt = "今天天气真好"
    generated = trainer.generate(prompt)
    print(f"提示: {prompt}")
    print(f"生成: {generated}")

if __name__ == '__main__':
    main() 