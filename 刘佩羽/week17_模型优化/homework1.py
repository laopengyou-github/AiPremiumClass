import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

def get_batch(split):
    # 选择训练或验证数据集
    data = train_data if split == 'train' else val_data

    # 动态从数据集中选择位置索引
    ix = torch.randint(len(data) - block_size, (batch_size,)) # [0,103846]随机生成位置索引，向后截取block_size字符训练
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])

    return x.to(device),y.to(device)

class Block(nn.Module):

    def __init__(self, n_embd, n_head, dropout):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, n_embd, head_size, dropout)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))   # 残差连接
        x = x + self.ffwd(self.ln2(x)) # 残差连接
        return x

class FeedFoward(nn.Module):

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, n_embd*4),
            nn.ReLU(),
            nn.Linear(4*n_embd, n_embd),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, n_embd, head_embd, dropout):
        super().__init__()
        self.heads = nn.ModuleList([Head(n_embd, head_embd, dropout) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        return out

class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, n_embd, head_embd, dropout):
        super().__init__()
        self.key = nn.Linear(n_embd, head_embd, bias=False)
        self.query = nn.Linear(n_embd, head_embd, bias=False)
        self.value = nn.Linear(n_embd, head_embd, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, input_x):
        # B, T, C = input_x.shape
        k = self.key(input_x)   # (B, T, head_size)
        q = self.query(input_x) # (B, T, head_size)
        v = self.value(input_x) # (B, T, 16)

        # wei = q @ k.transpose(-2,-1) * C ** -0.5

        # T = wei.shape[-1]
        # tril = torch.tril(torch.ones(T, T, device=wei.device))
        # wei = wei.masked_fill(tril == 0, float('-inf'))
        # wei = wei.softmax(dim=-1)
        # wei = self.dropout(wei)
        
        # out = wei @ v
        # return out

        ##################### flash attention 2 #########################

        # FlashAttention2 只支持bfloat16或float16类型的张量
        q = q.to(torch.bfloat16)
        k = k.to(torch.bfloat16)
        v = v.to(torch.bfloat16) # 或调用 v.bfloat16()

        with sdpa_kernel(backends=SDPBackend.FLASH_ATTENTION):
            attn_output = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        return attn_output


class BingramLanguageModel(nn.Module):
    2
    def __init__(self, block_size, vocab_size, n_embd, n_head, n_layer, dropout):
        super().__init__()
        # 每个token直接输出的logits值作为下一个token的映射
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)

        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head, dropout=dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer normalization
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx和target都是维度为 (B,T) 的整型tensor
        tok_emb = self.token_embedding_table(idx)  # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device), ) # (T,C)
        x = tok_emb + pos_emb  # (B, T, C)
        x = self.blocks(x)  # (B, T, C)
        x = self.ln_f(x)
        logits = self.lm_head(x)  # (B,T, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.reshape(B * T, C)
            targets = targets.reshape(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        # idx指当前语料集(B,T)中的索引
        for _ in range(max_new_tokens):
            # 限定索引列的取值范围
            idx_cond = idx[:, -block_size:]
            # 推理
            logits, loss = self(idx_cond)
            # 只提取最后一个时间步的结果
            logits = logits[:, -1, :]  # (B,C)
            # 通过softmax转换为概率值
            probs = F.softmax(logits, dim=-1)  # (B,C)
            # 随机采样
            idx_next = torch.multinomial(probs, num_samples=1)  # (B,1)
            # 把采样的索引追加在当前解码序列末尾
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx

if __name__ == '__main__':

    # 模型训练数据集
    block_size = 8
    batch_size = 32
    max_iter = 5000
    learn_rate = 1e-3
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    n_embd = 32
    eval_interval = 500
    eval_iters = 200
    head_size = 8
    num_layers = 4
    dropout = 0.1

    
    with open('input.txt', encoding='utf-8') as f:
        text = f.read()

    # 字典、编码器(函数)、解码器(函数)
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    stoi = {ch:i for i,ch in enumerate(chars)}  #str_to_index
    itos = {i:ch for i,ch in enumerate(chars)}  #index_to_str

    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])

    # 文本转换token index
    data = torch.tensor(encode(text), dtype=torch.long)

    # 拆分数据集
    n = int(len(data) * .9)
    train_data = data[:n]
    val_data = data[n:]

    # 模型训练
    model = BingramLanguageModel(block_size, vocab_size, n_embd, head_size, num_layers, dropout)
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

        # 批次样本
        xb, yb = get_batch('train')

        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    # 模型生成
    idx = torch.zeros((1,1), dtype=torch.long, device=device)
    print(decode(model.generate(idx, max_new_tokens=500)[0].tolist())) 