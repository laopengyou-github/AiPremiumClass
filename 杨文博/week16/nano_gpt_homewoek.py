# -*- coding: utf-8 -*-
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer

class ChineseDataset(Dataset):
    def __init__(self, file_path, tokenizer, block_size=128):
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()

        self.encodings = tokenizer.encode(text)
        self.block_size = block_size
        self.tokenizer = tokenizer
    
    def __len__(self):
        return len(self.encodings) - self.block_size
    
    def __getitem__(self, idx):
        return torch.tensor(self.encodings[idx:idx+self.block_size], dtype=torch.long)

class NanoGPT(nn.Module):
    def __init__(self, vocab_size, n_embd=256, n_head=8, n_layer=6, block_size=128):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)
        
        self.blocks = nn.Sequential(*[
            nn.TransformerEncoderLayer(d_model=n_embd, nhead=n_head)
            for _ in range(n_layer)
        ])
        
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.block_size = block_size
    
    def forward(self, x):
        b, t = x.size()
        assert t <= self.block_size, "输入序列过长"

        tok_emb = self.token_embedding(x)
        pos = torch.arange(0, t, dtype=torch.long, device=x.device).unsqueeze(0)
        pos_emb = self.position_embedding(pos)
        x = tok_emb + pos_emb

        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits
    
    def generate(self, prompt, max_new_tokens=50, temperature=1.0):
        self.eval()
        with torch.no_grad():
            input_ids = prompt
            
            for _ in range(max_new_tokens):

                input_ids_trunc = input_ids if input_ids.size(1) <= self.block_size else input_ids[:, -self.block_size:]

                logits = self(input_ids_trunc)

                logits = logits[:, -1, :] / temperature

                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

                input_ids = torch.cat([input_ids, next_token], dim=1)
            
            return input_ids

def train_nano_gpt():
    # ==================== 1. 准备数据 ====================
    # 创建示例中文语料（实际使用时替换为自己的文本文件）
    os.makedirs("data/chinese", exist_ok=True)
    if not os.path.exists("data/chinese/input.txt"):
        sample_text = """人工智能是当今科技发展的前沿领域。深度学习技术近年来取得了显著进展。
        自然语言处理使计算机能够理解和生成人类语言。计算机视觉让机器能够看懂世界。
        强化学习通过试错机制让AI系统自主学习。大语言模型如GPT系列展现了惊人的文本生成能力。"""
        
        with open("data/chinese/input.txt", "w", encoding="utf-8") as f:
            f.write(sample_text * 50)  
    
    tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
    dataset = ChineseDataset("data/chinese/input.txt", tokenizer)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NanoGPT(
        vocab_size=tokenizer.vocab_size,
        n_embd=256,
        n_head=8,
        n_layer=4,
        block_size=128
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=6e-4)
    criterion = nn.CrossEntropyLoss()

    print("开始训练nanoGPT中文模型...")
    for epoch in range(10):  # 示例用10个epoch
        total_loss = 0
        for batch in dataloader:
            batch = batch.to(device)
            
            optimizer.zero_grad()
            logits = model(batch[:, :-1])
            loss = criterion(logits.reshape(-1, tokenizer.vocab_size), batch[:, 1:].reshape(-1))
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(dataloader):.4f}")

    torch.save(model.state_dict(), "nano_gpt_chinese.pt")

    def generate_text(prompt, max_length=50, temperature=0.7):
        model.eval()
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        
        generated = model.generate(
            input_ids,
            max_new_tokens=max_length,
            temperature=temperature
        )
        
        return tokenizer.decode(generated[0].tolist(), skip_special_tokens=True)

    test_prompts = [
        "人工智能的未来",
        "深度学习可以应用于",
        "自然语言处理技术"
    ]
    
    print("\n" + "="*60)
    print("nanoGPT中文生成测试")
    print("="*60)
    
    for prompt in test_prompts:
        generated = generate_text(prompt)
        print(f"输入: {prompt}")
        print(f"生成: {generated}")
        print("-"*50)

if __name__ == "__main__":
    train_nano_gpt()
