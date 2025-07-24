import torch
from dotenv import load_dotenv, find_dotenv
import torch.nn as nn
import torch.nn.functional as F


def get_batch_data(split):
    data = train_data if split == 'train' else val_data
    idx = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in idx])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in idx])
    return x.to(device), y.to(device)


class Head(nn.Module):
    def __init__(self, n_emd):
        super().__init__()
        self.key = nn.Linear(n_emd, n_emd, bias=False)
        self.query = nn.Linear(n_emd, n_emd, bias=False)
        self.value = nn.Linear(n_emd, n_emd, bias=False)

    def forward(self, input_x):
        B, T, C = input_x.shape

        k = self.key(input_x)
        q = self.query(input_x)
        v = self.value(input_x)

        attn = q @ k.transpose(-2, -1) * C ** -0.5
        T = attn.shape[1]
        tril = torch.tril(torch.ones(T, T, device=device))
        mask = attn.masked_fill(tril == 0, float('-inf'))
        wei = mask.softmax(dim=-1)

        out = wei @ v
        return out


class BingramLanguageModel(nn.Module):

    def __init__(self, vocab_size, block_size, n_emd):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_emd)
        self.position_token_embedding_table = nn.Embedding(block_size, n_emd)
        self.head = Head(n_emd)
        self.lm_head = nn.Linear(n_emd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        token_emd = self.token_embedding_table(idx)
        position_emd = self.position_token_embedding_table(torch.arange(T, device=device))
        emd = token_emd + position_emd
        x = self.head(emd)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx (B,T) 数组对应着当前的输入内容 [1,1]
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            # 模型推理
            logits, loss = self.forward(idx_cond)
            # 获取最后一个时间步的输出
            logits = logits[:, -1, :]
            # 应用softmax转换为概率值
            probs = F.softmax(logits, dim=-1)
            # 按权重值采样，返回对应的索引
            # idx_next = torch.argmax(probs, dim=-1)
            # 随机采样
            idx_next = torch.multinomial(probs, num_samples=1)
            # 应用采样后的索引
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


if __name__ == '__main__':
    load_dotenv(find_dotenv())

    # 构建训练数据集
    block_size = 8
    batch_size = 4
    max_iter = 5000
    learn_rate = 1e-3
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_embd = 32
    eval_interval = 500
    eval_iters = 200

    with open('急难愁盼文件.txt', 'r', encoding='utf-8') as f:
        text = f.read()

    # 词典
    vocab = sorted(list(set(text)))
    vocab_size = len(vocab)

    # 编码器
    str_to_idx = {s: i for i, s in enumerate(vocab)}
    # 解码器
    idx_to_str = {i: s for i, s in enumerate(vocab)}

    # 编码方法
    encode = lambda txt: [str_to_idx[s] for s in txt]
    # 解码方法
    decode = lambda idx_list: ''.join([idx_to_str[idx] for idx in idx_list])

    # 文本转换为索引
    data = torch.tensor(encode(text), dtype=torch.long)
    # 拆分数据
    split = int(0.9 * len(data))
    train_data = data[:split]
    val_data = data[split:]

    # 引入大模型
    model = BingramLanguageModel(vocab_size, block_size, n_embd)
    model.to(device)


    @torch.no_grad()
    def estimate_loss():
        out = {}
        model.eval()
        for split in ['train', 'val']:
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                X, Y = get_batch_data(split)
                logits, loss = model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        model.train()
        return out


    # 模型训练、优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for iter in range(1000):
        if iter % eval_interval == 0:
            losses = estimate_loss()
            print(f"step: {iter}, train loss: {losses['train']:.4f}, val loss: {losses['val']:.4f}")

        xb, yb = get_batch_data('train')
        logits, loss = model(xb, yb)
        loss.backward()
        optimizer.step()
        model.zero_grad()

    # 模型推理
    # 模型推理
    token_idx = torch.zeros((1, 1), dtype=torch.long, device=device)
    result = model.generate(token_idx, 500)
    print(decode(result[0].tolist()))
