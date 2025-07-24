'''
2. 使用nano-gpt2.0模型，训练一份中文语料并测试生成。
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

# nanoGPT核心结构 
class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd, n_head, block_size):
        super().__init__()
        self.n_head = n_head
        self.head_size = n_embd // n_head
        self.key = nn.Linear(n_embd, n_embd, bias=False)
        self.query = nn.Linear(n_embd, n_embd, bias=False)
        self.value = nn.Linear(n_embd, n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x).view(B, T, self.n_head, self.head_size).transpose(1,2)  # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, self.head_size).transpose(1,2)  # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, self.head_size).transpose(1,2)  # (B, nh, T, hs)
        att = (q @ k.transpose(-2, -1)) * (1.0 / (self.head_size ** 0.5))  # (B, nh, T, T)
        att = att.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        out = att @ v  # (B, nh, T, hs)
        out = out.transpose(1,2).contiguous().view(B, T, C)  # (B, T, C)
        out = self.proj(out)
        return out

class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
        )
    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embd, n_head, block_size):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.attn = MultiHeadAttention(n_embd, n_head, block_size)
        self.ffwd = FeedForward(n_embd)
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class NanoGPT2(nn.Module):
    def __init__(self, block_size, vocab_size, n_embd, n_head, n_layer):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head, block_size) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.block_size = block_size
    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)  # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))  # (T,C)
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)  # (B,T,vocab_size)
        loss = None
        if targets is not None:
            logits_flat = logits.view(-1, logits.size(-1))
            targets_flat = targets.view(-1)
            loss = F.cross_entropy(logits_flat, targets_flat)
        return logits, loss
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# 数据加载和训练流程 
if __name__ == '__main__':
    block_size = 64  # 建议增大block_size以适配transformer
    batch_size = 32
    max_iter = 5000
    learn_rate = 1e-3
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    n_embd = 128
    n_head = 4
    n_layer = 4
    eval_interval = 500
    eval_iters = 200

    with open('./陈兴/week16/homework2/西游记.txt') as f:
        text = f.read()
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    stoi = {ch:i for i,ch in enumerate(chars)}
    itos = {i:ch for i,ch in enumerate(chars)}
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
    data = torch.tensor(encode(text), dtype=torch.long)
    n = int(len(data) * .9)
    train_data = data[:n]
    val_data = data[n:]

    def get_batch(split):
        data_ = train_data if split == 'train' else val_data
        ix = torch.randint(len(data_) - block_size, (batch_size,))
        x = torch.stack([data_[i:i+block_size] for i in ix])
        y = torch.stack([data_[i+1:i+block_size+1] for i in ix])
        return x.to(device), y.to(device)

    model = NanoGPT2(block_size, vocab_size, n_embd, n_head, n_layer)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learn_rate)

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

    for iter in range(max_iter):
        if iter % eval_interval == 0:
            losses = estimate_loss()
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        xb, yb = get_batch('train')
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    idx = torch.zeros((1,1), dtype=torch.long, device=device)
    print(decode(model.generate(idx, max_new_tokens=500)[0].tolist())) 