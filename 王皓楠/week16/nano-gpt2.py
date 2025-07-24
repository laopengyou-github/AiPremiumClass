import torch
import torch.nn as nn
import torch.nn.functional as F

# 全局参数（中文训练可适当调整）
block_size = 32  # 中文句子较长，可适当增大上下文窗口
batch_size = 32
max_iter = 10000  # 中文语料通常需要更多迭代
learn_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
n_embd = 64  # 中文语义更复杂，可适当增大嵌入维度
eval_interval = 500
eval_iters = 200


def get_batch(split, train_data, val_data, block_size, batch_size, device):
    """获取训练/验证批次数据（修改为参数传入，避免依赖全局变量）"""
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)


class Head(nn.Module):
    """单头自注意力（无需修改）"""
    def __init__(self, n_embd):
        super().__init__()
        self.key = nn.Linear(n_embd, n_embd, bias=False)
        self.query = nn.Linear(n_embd, n_embd, bias=False)
        self.value = nn.Linear(n_embd, n_embd, bias=False)

    def forward(self, input_x):
        B, T, C = input_x.shape
        k = self.key(input_x)  # (B,T,C)
        q = self.query(input_x)  # (B,T,C)
        wei = q @ k.transpose(-2, -1) * C **-0.5  # (B,T,T)
        tril = torch.tril(torch.ones(T, T, device=device))  # 下三角掩码（中文同样需要因果掩码）
        wei = wei.masked_fill(tril == 0, float('-inf'))
        wei = wei.softmax(dim=-1)
        v = self.value(input_x)  # (B,T,C)
        out = wei @ v  # (B,T,C)
        return out


class BingramLanguageModel(nn.Module):
    """二元语言模型（无需修改，适配任意字符集）"""
    def __init__(self, block_size, vocab_size, n_embd):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.sa_head = Head(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)  # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))  # (T,C)
        x = tok_emb + pos_emb  # (B,T,C)
        x = self.sa_head(x)  # (B,T,C)
        logits = self.lm_head(x)  # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(-1)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens, block_size):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]  # 截断到最大上下文长度
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]  # 取最后一个时间步
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


if __name__ == '__main__':
    # --------------------------
    # 1. 读取中文文本（关键修改）
    # --------------------------
    # 读取中文文件，指定编码为utf-8（避免中文乱码）
    with open('xi_you.txt', 'r', encoding='utf-8') as f:
        text = f.read()  # text为字符串，包含所有中文文本

    # --------------------------
    # 2. 构建中文字符集（关键修改）
    # --------------------------
    # 提取所有独特的字符（中文字符、标点、空格等）
    chars = sorted(list(set(text)))
    vocab_size = len(chars)  # 中文字符集通常比英文大（常见字约3000-5000）
    print(f"中文字符集大小: {vocab_size}")  # 可查看字符集规模

    # --------------------------
    # 3. 构建编解码器（关键修改）
    # --------------------------
    # 字符→索引映射（每个中文字符对应一个唯一索引）
    stoi = {ch: i for i, ch in enumerate(chars)}
    # 索引→字符映射（解码用）
    itos = {i: ch for i, ch in enumerate(chars)}

    # 编码函数：将中文文本（字符串）转换为索引列表
    def encode(s):
        return [stoi[ch] for ch in s]  # 遍历每个字符，转换为索引

    # 解码函数：将索引列表转换为中文文本
    def decode(l):
        return ''.join([itos[i] for i in l])  # 遍历每个索引，转换为字符并拼接

    # --------------------------
    # 4. 数据处理（与英文逻辑一致，适配中文）
    # --------------------------
    # 将文本转换为索引张量
    data = torch.tensor(encode(text), dtype=torch.long)
    # 拆分训练集和验证集
    n = int(len(data) * 0.9)
    train_data = data[:n]
    val_data = data[n:]

    # --------------------------
    # 5. 模型训练
    # --------------------------
    model = BingramLanguageModel(block_size, vocab_size, n_embd)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learn_rate)

    @torch.no_grad()
    def estimate_loss():
        out = {}
        model.eval()
        for split in ['train', 'val']:
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                # 传入参数，避免依赖全局变量
                X, Y = get_batch(split, train_data, val_data, block_size, batch_size, device)
                logits, loss = model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        model.train()
        return out

    for iter in range(max_iter):
        if iter % eval_interval == 0:
            losses = estimate_loss()
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        xb, yb = get_batch('train', train_data, val_data, block_size, batch_size, device)
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()


    idx = torch.zeros((1, 1), dtype=torch.long, device=device)
    # 生成500个中文字符
    generated = model.generate(idx, max_new_tokens=500, block_size=block_size)
    # 解码并打印结果
    print(decode(generated[0].tolist()))