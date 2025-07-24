from torch import nn
from torch.nn import functional as F
import torch

from 正则.练习正则 import result


class BingramLanguangeModel(nn.Module):
    def __init__(self,vocab_size):
        super().__init__()
        self.token_embeddings_table = nn.Embedding(vocab_size, vocab_size, padding_idx=0)

    def forward(self,idx,targets=None):
        logits = self.token_embeddings_table(idx)
        if targets is None:
            loss = None
        else:
            B,T,C = logits.shape
            logits = logits.view(-1, C)
            targets = targets.view(-1)
            loss = F.cross_entropy(logits, targets)
        return logits,loss

    def generate(self,idx,max_new_tokens):
        for _ in range(max_new_tokens):
            logits,loss = self(idx)
            logits = logits[:,-1,:]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, 1)
            idx = torch.cat((idx,idx_next),dim=1)
        return idx



if __name__=='__main__':

    with open('../网游之命轮之主_命给你行不行.txt','r',encoding='utf-8') as f:
        text = f.read()

    chat = sorted(list(set(text)))
    vocab_size = len(chat)

    str_to_index = {i:num for num,i in enumerate(chat)}
    index_to_str = {num:i for num,i in enumerate(chat)}

    encoder = lambda x : [str_to_index[i] for i in x]
    decoder = lambda x : ''.join([index_to_str[i] for i in x])

    data_ = torch.tensor(encoder(text),dtype=torch.long)

    num_data = int(len(data_)*0.8)

    train_data = data_[:num_data]
    valid_data = data_[num_data:]

    block_size = 8
    batch_size = 4


    def get_valid_batch(split):
        train_or_valid = train_data if split == 'train' else valid_data
        data = torch.randint(len(train_or_valid) - block_size, (batch_size,))
        x = torch.stack([train_or_valid[i:i + batch_size] for i in data])
        y = torch.stack([train_or_valid[i + 1:i + batch_size + 1] for i in data])
        return x, y


    #模型训练的创建
    model = BingramLanguangeModel(vocab_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    batch_size1 = 32

    for epoch in range(1000):
        xb, yb = get_valid_batch('train')
        logits,loss = model(xb,yb)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if epoch % 250 == 0:
            print(f'epoch:{epoch}\t,loss:{loss.item()}')

    token_index = torch.zeros((1,1),dtype=torch.long)
    result = model.generate(token_index,100)
    print(decoder(result[0].tolist()))

