import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import jieba 

# --- 1. 调整超参数 (Adjusted Hyperparameters) ---
# 建议为中文任务增加模型容量和上下文
block_size = 64      # 上下文长度，对于中文建议更长一些
batch_size = 64      # 批次大小
max_iter = 5000
learn_rate = 3e-4    # 对于更深的模型，使用稍小的学习率
device = 'cuda' if torch.cuda.is_available() else 'cpu'
n_embd = 128         # 嵌入维度，增加表示能力
n_head = 4           # 注意力头的数量
n_layer = 4          # Transformer Block 的层数
dropout = 0.2        # Dropout比例
eval_interval = 500
eval_iters = 200
# -------------------------------------------------

# 加载数据
file_path = os.path.join(os.path.dirname(__file__), 'input.txt')
with open(file_path, encoding='utf-8') as f:
    text = f.read()

# --- 使用 jieba 进行分词并构建词汇表 ---
print("正在使用jieba进行分词...")
# 使用jieba.lcut直接返回一个词语列表
words = jieba.lcut(text) 
print(f"分词完成，总词数: {len(words)}")

# 构建词汇表
vocab = sorted(list(set(words)))
# 新增：加入一个未知词标记
if '<UNK>' not in vocab:
    vocab.append('<UNK>')
vocab_size = len(vocab)
print(f"词汇表大小: {vocab_size}")

# 创建词语到索引、索引到词语的映射
stoi = {word:i for i,word in enumerate(vocab)}
itos = {i:word for i,word in enumerate(vocab)}

# 获取<UNK>标记的索引
UNK_IDX = stoi['<UNK>']

# 更新encode和decode函数
encode = lambda s: [stoi.get(word, UNK_IDX) for word in jieba.lcut(s)]
decode = lambda l: ''.join([itos[i] for i in l])

# --- 文本转换token index ---
data = torch.tensor([stoi[w] for w in words], dtype=torch.long)

n = int(len(data) * .8) # 使用90%作为训练集
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class Head(nn.Module):
    """ 单头 self-attention (与您原版一致, 但增加了dropout) """
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        # 新增：注册一个非参数的tril缓冲，避免每次都创建
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        # 新增：Dropout层
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2,-1) * k.size(-1)**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        # 新增：应用dropout
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out

# --- 2. 新增：多头注意力机制 (MultiHeadAttention) ---
class MultiHeadAttention(nn.Module):
    """ 将多个Head并行的多头注意力 """
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        # 新增一个线性层，用于在拼接后进行投影
        self.proj = nn.Linear(n_embd, n_embd)
        # 新增：Dropout层
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # 将所有头的输出拼接起来
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        # 应用投影和dropout
        out = self.dropout(self.proj(out))
        return out

# --- 3. 新增：前馈网络 (FeedForward) ---
class FeedForward(nn.Module):
    """ 一个简单的线性层 + ReLU非线性激活函数 """
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd), # 按照论文标准，中间层维度是4倍
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd), # 投影回原始维度
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

# --- 4. 新增：Transformer核心模块 (Block) ---
class Block(nn.Module):
    """ Transformer block: communication followed by computation """
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        # 新增：层归一化 (Layer Normalization)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        # 残差连接 (x + ...)
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

# --- 5. 改造后的语言模型 ---
class GPTLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        # 修改：使用堆叠的Block代替原来的单头注意力
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        # 新增：在最后的blocks后加上LayerNorm
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        # 修改：通过所有Block
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# --- 模型训练 ---
model = GPTLanguageModel()
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learn_rate)

for iter in range(max_iter):
    if iter % eval_interval == 0 or iter == max_iter - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# --- 模型生成 ---
idx = torch.zeros((1,1), dtype=torch.long, device=device)
print(decode(model.generate(idx, max_new_tokens=500)[0].tolist()))