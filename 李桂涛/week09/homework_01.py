import torch
import torch.nn as nn
import torch.optim as optim
import math
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.nn.utils.rnn import pad_sequence
from collections import Counter
import numpy as np

# 位置编码矩阵
class PositionalEncoding(nn.Module):
    def __init__(self, emb_size, dropout, maxlen=5000):
        super().__init__()
        # embedding维度 偶数列的缩放指数(行缩放指数值)
        den = torch.exp(- torch.arange(0,emb_size,2) * math.log(10000)/emb_size)
        # 生成位置编码索引 5000 --> (5000,1)
        pos = torch.arange(0, maxlen).reshape(maxlen,1) 
        # 编码矩阵 (5000,1) --> (5000, emb_size)      
        pos_embdding = torch.zeros((maxlen,emb_size))
        #为了使位置编码能够捕捉到位置之间的相对关系，并且能够在不同的维度上区分不同的位置
        pos_embdding[:,0::2] =  torch.sin(pos * den)  #(5000, emb_size) pos_embedding 全是0，偶数列
        pos_embdding[:,1::2] =  torch.cos(pos * den)  #(5000, emb_size) pos_embedding 一半是0，奇数列
        pos_embdding = pos_embdding.unsqueeze(0)  # 添加和batch对应维度 (1, 5000, emb_size) unqueeze(-2)==unsqueeze(0)
        # dropout
        self.dropout = nn.Dropout(dropout)
        # 注册当前矩阵不参与参数更新 位置编码不要更新！
        self.register_buffer('pos_embedding', pos_embdding) #将局部变量pos_embdding注册为了模块的缓冲区，并将其命名为 pos_embedding
        # register_buffer作用是将张量注册为模块的一个属性，必须通过 self.属性名来访问，
        # 与普通的局部变量不同，局部变量在方法结束后就不存在了，注册到缓冲区的变量会成为模块的持久属性
        
    def forward(self, token_embdding): # token_embdding (batch_size, token_len, emb_size)
        token_len = token_embdding.size(1)  # token长度 : token_len
        # (1, token_len, emb_size)
        add_emb = self.pos_embedding[:, :token_len, :] + token_embdding  #截取对应大小尺寸的位置编码添加到原本的矩阵中，这样就有位置概念了
        return self.dropout(add_emb)  #防止过拟合
    
class Seq2SeqTransform(nn.Module):
    #  d_model ：词转换的维度，也就是emd嵌入维度。在 Transformer 中，所有子层的输入和输出都保持这个维度
    #  nhead：表示多头注意力机制中的头数,num_enc_layers：表示编码器的层数,num_dec_layers：表示解码器的层数
    #  dim_forward：表示前向传播的维度，通常是多头注意力机制的维度,enc_voc_size：表示编码器词典的大小，dec_voc_size：表示解码器词典的大小
    def __init__(self,d_model,nhead,num_enc_layers,num_dec_layers,dim_forward,dropout,enc_voc_size,dec_voc_size):
        super().__init__()
        #transformer
        self.transformer = nn.Transformer(
            d_model=d_model,                   #嵌入维度
            nhead=nhead,                       #多头注意力机制个数
            num_encoder_layers=num_enc_layers, #encoder层数
            num_decoder_layers=num_dec_layers, 
            dim_feedforward=dim_forward,       #前馈神经网络的维度
            dropout=dropout,
            batch_first=True                   #输入数据的维度(batch_size, seq_len, emb_size)
        )
        #encoder input embedding     [batch_size, seq_len] ，经过嵌入层后，输出形状变为 [batch_size, seq_len, d_model]
        self.enc_emb = nn.Embedding(enc_voc_size, d_model)
        #decoder input embedding
        self.dec_emb = nn.Embedding(dec_voc_size, d_model)
        # predict generate linear
        self.predict = nn.Linear(d_model, dec_voc_size)  # token预测基于解码器词典
        self.pos_encoding = PositionalEncoding(d_model,dropout)
        
    def forward(self, enc_inp, dec_inp, tgt_mask, enc_pad_mask, dec_pad_mask):
        #enc_inp 和 dec_inp ：分别表示编码器和解码器的输入序列
        # multi head attention之前基于位置编码embedding生成
        enc_emb = self.pos_encoding(self.enc_emb(enc_inp))
        dec_emb = self.pos_encoding(self.dec_emb(dec_inp))
        # 调用transformer计算
        outs = self.transformer(src=enc_emb, tgt=dec_emb, tgt_mask=tgt_mask,
                         src_key_padding_mask=enc_pad_mask, 
                         tgt_key_padding_mask=dec_pad_mask)
        # 推理
        return self.predict(outs)
    
    # 推理环节使用方法
    def encode(self, enc_inp):
        enc_emb = self.pos_encoding(self.enc_emb(enc_inp))
        return self.transformer.encoder(enc_emb)
    
    def decode(self, dec_inp, memory, dec_mask):
        dec_emb = self.pos_encoding(self.dec_emb(dec_inp))
        return self.transformer.decoder(dec_emb, memory, dec_mask)

# 假设数据集文件为 'couplets.txt'，格式为每行“上联|下联”
def load_data(file_path):
    enc_sentences, dec_sentences = [], []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('|')
            if len(parts) == 2:
                enc_sentences.append(parts[0])
                dec_sentences.append(parts[1])
    return enc_sentences, dec_sentences

enc_sentences, dec_sentences = load_data('D:/LGT_Private/pythonProject/bd/w9/资料/hw/couplets.txt')

# 构建词汇表
def build_vocab(sentences, special_tokens=('<pad>', '<sos>', '<eos>', '<unk>')):
    vocab = {token: idx for idx, token in enumerate(special_tokens)}
    counter = Counter()
    for sent in sentences:
        counter.update(sent)
    for char, _ in counter.most_common():
        if char not in vocab:
            vocab[char] = len(vocab)
    return vocab

enc_vocab = build_vocab(enc_sentences)
dec_vocab = build_vocab(dec_sentences)
enc_vocab_size = len(enc_vocab)
dec_vocab_size = len(dec_vocab)

# 将文本转换为索引序列
def text_to_indices(sentences, vocab, add_special_tokens=False):
    indices_list = []
    for sent in sentences:
        indices = [vocab['<sos>']] if add_special_tokens else []
        for char in sent:
            indices.append(vocab.get(char, vocab['<unk>']))
        if add_special_tokens:
            indices.append(vocab['<eos>'])
        indices_list.append(torch.tensor(indices, dtype=torch.long))
    return indices_list

enc_data = text_to_indices(enc_sentences, enc_vocab)
dec_data = text_to_indices(dec_sentences, dec_vocab, add_special_tokens=True)

class CoupletsDataset(Dataset):
    def __init__(self, enc_data, dec_data):
        self.enc_data = enc_data
        self.dec_data = dec_data

    def __len__(self):
        return len(self.enc_data)

    def __getitem__(self, idx):
        return self.enc_data[idx], self.dec_data[idx]

def collate_fn(batch):
    enc_batch, dec_batch = zip(*batch)
    enc_padded = pad_sequence(enc_batch, batch_first=True, padding_value=0)
    dec_full_padded = pad_sequence(dec_batch, batch_first=True, padding_value=0)
    dec_input = dec_full_padded[:, :-1]
    dec_output = dec_full_padded[:, 1:]
    tgt_len = dec_input.size(1)
    tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_len)
    return {
        'enc_input': enc_padded,
        'dec_input': dec_input,
        'dec_output': dec_output,
        'enc_pad_mask': (enc_padded == 0),
        'dec_pad_mask': (dec_input == 0),
        'tgt_mask': tgt_mask
    }

dataset = CoupletsDataset(enc_data, dec_data)
dataloader = DataLoader(dataset, batch_size=6, shuffle=True, collate_fn=collate_fn)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = Seq2SeqTransform(
    d_model=256,
    nhead=8,
    num_enc_layers=3,
    num_dec_layers=3,
    dim_forward=1024,
    dropout=0.1,
    enc_voc_size=enc_vocab_size,
    dec_voc_size=dec_vocab_size
).to(device)

optimizer = optim.Adam(model.parameters(), lr=0.0001)
criterion = nn.CrossEntropyLoss(ignore_index=0)

num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch in dataloader:
        enc_input = batch['enc_input'].to(device)
        dec_input = batch['dec_input'].to(device)
        dec_output = batch['dec_output'].to(device)
        enc_pad_mask = batch['enc_pad_mask'].to(device)
        dec_pad_mask = batch['dec_pad_mask'].to(device)
        tgt_mask = batch['tgt_mask'].to(device)

        optimizer.zero_grad()
        output = model(enc_input, dec_input, tgt_mask, enc_pad_mask, dec_pad_mask)
        # loss = criterion(output.view(-1, dec_vocab_size), dec_output.view(-1))
        loss = criterion(output.reshape(-1, dec_vocab_size), dec_output.reshape(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')

# 保存模型
torch.save(model.state_dict(), 'transformer_couplet_model.pth')