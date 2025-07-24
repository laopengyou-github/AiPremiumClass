import torch
import torch.nn as nn
import torch.nn.functional as F
import jieba
 
def get_batch(split):
   
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,)) 
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
 
    return x.to(device),y.to(device)
 
class Head(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.key = nn.Linear(n_embd, n_embd, bias=False)
        self.query = nn.Linear(n_embd, n_embd, bias=False)
        self.value = nn.Linear(n_embd, n_embd, bias=False)
 
    def forward(self, input_x):
        B, T, C = input_x.shape
 
        k = self.key(input_x)
        q = self.query(input_x)
        v = self.value(input_x)
 
        wei = q @ k.transpose(-2,-1) * C ** -0.5
        T = wei.shape[-1]
        tril = torch.tril(torch.ones(T,T, device=device))
        wei = wei.masked_fill(tril == 0, float('-inf'))
        wei = wei.softmax(dim=-1)
 
        out = wei @ v
        return out
 
 
class BingramLanguageModel(nn.Module):
     
    def __init__(self, block_size, vocab_size, n_embd):
        super().__init__()
       
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
 
        self.sa_head = Head(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
         
    def forward(self, idx, targets=None):
        B,T = idx.shape
         
        tok_emb = self.token_embedding_table(idx)  
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        x = self.sa_head(x)
        logits = self.lm_head(x)
         
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(-1)
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
 
if __name__ == '__main__':
 
   
    block_size = 16  
    batch_size = 32
    max_iter = 100000
    base_lr = 1e-3
    min_lr = 1e-4
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    n_embd = 64  
    eval_interval = 1000
    eval_iters = 200
 
     
    # with open('input.txt') as f:
    #     text = f.read()
    with open('input.txt', encoding='utf-8') as f:
        text = f.read()
 

    # 使用jieba分词构建词表
    words = list(jieba.cut(text))
    vocab = sorted(list(set(words)))
    vocab_size = len(vocab)
    stoi = {w: i for i, w in enumerate(vocab)}  # word to index
    itos = {i: w for i, w in enumerate(vocab)}  # index to word

    encode = lambda s: [stoi[w] for w in jieba.cut(s)]
    decode = lambda l: ''.join([itos[i] for i in l])
 

    # 文本转换为token index
    data = torch.tensor(encode(text), dtype=torch.long)
 
    # 拆分数据集
    n = int(len(data) * .9)
    train_data = data[:n]
    val_data = data[n:]
 
    # 模型训练
    model = BingramLanguageModel(block_size, vocab_size, n_embd)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr)
    # 使用余弦退火动态学习率
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_iter, eta_min=min_lr)
 
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
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}, lr {optimizer.param_groups[0]['lr']:.6f}")

        # 批次样本
        xb, yb = get_batch('train')

        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()
 
    # 模型生成
    idx = torch.zeros((1,1), dtype=torch.long, device=device)
    print(decode(model.generate(idx, max_new_tokens=1000)[0].tolist()))