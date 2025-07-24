import torch
import torch.nn as nn

from torch.nn import functional as F

torch.manual_seed(1337)


class BingramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)
        
    def forward(self, idx, targets=None):
        logits = self.token_embedding_table(idx)
        
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
            logits, loss = self(idx)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs,num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
        

def main():
    with open('./邪王真眼/week16/input.txt', 'r', encoding='utf-8') as f:
        text = f.read()

    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    stoi = {ch:i for i,ch in enumerate(chars)}
    itos = {i:ch for i,ch in enumerate(chars)}
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])

    data = torch.tensor(encode(text), dtype=torch.long)

    n = int(0.9*len(data))
    train_data = data[:n]
    val_data = data[n:]

    batch_size = 4
    block_size = 8
    def get_batch(split):
        data = train_data if split =='train' else val_data
        
        ix = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack([data[i:i+block_size] for i in ix]) # (B,T)->(4,8)
        y = torch.stack([data[i+1:i+block_size+1] for i in ix])
        return x,y
    
    xb, yb = get_batch('train')

    m = BingramLanguageModel(vocab_size)
    out, loss = m(xb, yb)
    optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)
    batch_size = 32
    for steps in range(10000):
        xb, yb = get_batch('train')
        logits, loss = m(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        
    print(loss.item())
    print(decode(m.generate(idx = torch.zeros((1,1), dtype=torch.long), max_new_tokens=500)[0].tolist()))


if __name__ == "__main__":
    main()
    