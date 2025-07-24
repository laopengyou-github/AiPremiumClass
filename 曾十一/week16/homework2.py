import os
import torch
import torch.nn as nn
import torch.nn.functional as F

# =============================================================================
# 1. 数据加载函数 (回到简单的 get_batch)
# =============================================================================
def get_batch(split, train_data, val_data, block_size, batch_size, device):
    """获取一个批次的数据。"""
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

# =============================================================================
# 2. 模型定义 (保持升级后的强大架构)
# =============================================================================
class Head(nn.Module):
    def __init__(self, n_embd, head_size, block_size, dropout):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)
        wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        head_size = n_embd // n_head
        self.heads = nn.ModuleList([Head(n_embd, head_size, block_size, dropout) for _ in range(n_head)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out

class FeedForward(nn.Module):
    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        self.sa = MultiHeadAttention(n_embd, n_head, block_size, dropout)
        self.ffwd = FeedForward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class GPTLanguageModel(nn.Module):
    def __init__(self, vocab_size, n_embd, block_size, n_head, n_layer, dropout):
        super().__init__()
        self.block_size = block_size
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head, block_size, dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        self.train()
        return idx

# =============================================================================
# 3. 主程序入口 (单卡训练逻辑)
# =============================================================================
if __name__ == '__main__':
    # --- 超参数配置 ---
    args = {
        'data_path': r'/mnt/data_1/zfy/self/八斗精品班/第十六周_GPT系列模型搭建训练及优化/homework/剑来(1-500章).txt',
        'block_size': 256,
        'batch_size': 128,  # 如果单卡显存不足，可以适当减小此值
        'max_iters': 5000, # 训练迭代次数
        'eval_interval': 500,
        'eval_iters': 200,
        'learn_rate': 3e-4,
        'n_embd': 384,
        'n_head': 6,
        'n_layer': 6,
        'dropout': 0.2 # 可以适当增加dropout率以防止单卡过拟合
    }

    # 设置要使用的单张GPU (例如 '0', '1', '2' ...)
    # 如果只有一张卡或者想用默认的第一张卡，可以注释掉这行
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # --- 数据准备 ---
    try:
        with open(args['data_path'], encoding='utf-8') as f:
            text = f.read()
    except FileNotFoundError:
        print(f"错误: 找不到语料文件 {args['data_path']}")
        exit()

    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
    
    data = torch.tensor(encode(text), dtype=torch.long)
    n = int(len(data) * 0.9)
    train_data = data[:n]
    val_data = data[n:]

    # --- 模型设置 ---
    model = GPTLanguageModel(
        vocab_size=vocab_size,
        n_embd=args['n_embd'],
        block_size=args['block_size'],
        n_head=args['n_head'],
        n_layer=args['n_layer'],
        dropout=args['dropout']
    ).to(device)
    
    print(f"模型参数量: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args['learn_rate'])

    # --- 评估函数 ---
    @torch.no_grad()
    def estimate_loss():
        out = {}
        model.eval()
        for split in ['train', 'val']:
            losses = torch.zeros(args['eval_iters'])
            for k in range(args['eval_iters']):
                X, Y = get_batch(split, train_data, val_data, args['block_size'], args['batch_size'], device)
                _, loss = model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        model.train()
        return out

    # --- 训练循环 ---
    for iter in range(args['max_iters']):
        if iter % args['eval_interval'] == 0 or iter == args['max_iters'] - 1:
            losses = estimate_loss()
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        xb, yb = get_batch('train', train_data, val_data, args['block_size'], args['batch_size'], device)
        
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    # --- 保存和生成 ---
    print("训练完成!")
    torch.save(model.state_dict(), 'gpt_model_single_gpu.pth')
    print("模型已保存至 gpt_model_single_gpu.pth")

    print("\n--- 开始生成文本 ---")
    start_context = torch.zeros((1, 1), dtype=torch.long, device=device)
    generated_indices = model.generate(start_context, max_new_tokens=500)[0].tolist()
    print(decode(generated_indices))
