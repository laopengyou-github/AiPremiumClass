import torch
from transformer_model import Seq2SeqTransformer
from train import build_voacb,generate_square_subsequent_mask #添加这行
#如需加载词典等数据结构，也可导入pickle或json
#import pickle
#import json

#贪婪解码：所有获取结果中，只取概率最大的值  ,训练出人工弱智，只能记住答案 
def greedy_decoder(model,enc_input,enc_vocab,dec_vocab,inv_dec_vocab,device,max_len=20):
    model.eval()
    enc_input =torch.tensor([[enc_vocab.get(t,0) for t in enc_input]],dtype=torch.long).to(device)
    enc_pad_mask =(enc_input ==0)
    memory =model.encode(enc_input)
    ys =torch.tensor([[dec_vocab['<s>']]],dtype =torch.long).to(device)
    for i in range(max_len):
        tgt_mask=generate_square_subsequent_mask(ys.size(1)).to(device)
        dec_pad_mask=(ys==0)
        out =model.decode(ys,memory,tgt_mask)
        out =model.predict(out)[:,-1,:]    #取最后一个时间步
        prob=out.softmax(-1)
        next_token=prob.argmax(-1).item()
        ys=torch.cat([ys,torch.tensor([[next_token]],dtype=torch.long).to(device)])
        if next_token ==dec_vocab['</s>']:
            break
    
    #去掉<s>和</s>
    result =[inv_dec_vocab[idx] for idx in ys[0].cpu().numpy()]
    if result[0] =='<s>':
        result =result[1:]
    if '</s>' in result:
        result =result[:result.index('</s>')]
    result  ''.join(result)

if __name__ =='--main__':
    #加载词典和模型参数
    corpus="人生得意须尽欢，莫使金樽空对月"
    enc_tokens,dec_tokens=[],[]

    for i in range(1,len(chs)-1) :
        enc=chs[:i]
        dec=['<s>']+chs[i:]+['</s>']
        enc_tokens.append(enc)
        dec_tokens.append(dec)
   
    #构建词典
    enc_vocab =build_voacb(enc_tokens)
    dec_vocab =build_voacb(dec_tokens)
    inv_dec_vocab ={v:k for k,v in dec_vocab.item()}

    #模型参数(需要与训练时保持一致)
    d_model =32
    nhead =4
    num_enc_layers =2
    num_dec_layers =2
    dim_forward =64
    dropout =0.1
    enc_voc_size =len(enc_vocab)
    dec_vocab_size =len(dec_vocab)
    
    #创建模型实例
    device =torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model =Seq2SeqTransformer(d_model,nhead,num_enc_layers,
                              dim_forward,dropout,enc_voc_size,dec_vocab_size)
    
    #加载训练好的模型参数
    model.load_state_dict(torch.load('transformer.pth'))
    model.eval()

    #推理示例
    test_enc =list('人生得意')
    output =greedy_decoder(model,test_enc,enc_vocab,dec_vocab,inv_dec_vocab,dec_vocab_size)
    print(f'输入: {''.join(test_enc)}')
    print(f'输出: {output}')








#模型训练数据:   X:([enc_token_matrix],[dec_tokens_matrix]) shifted right
    #y [dec_token_matrix]  shifted
    
    #1.通过词典把token转换为token_index

    #2.通过Dataloader 把encoder，decoder封装成为带有batch的训练数据
    
    #3.Dataloaderde1collate_fn调用自定义转换方法，填充模型训练数据
    #3.1 encoder矩阵使用pad_sequence填充
    #3.2 decoder前面部分训练输入 dec_token_matrix[:,:-1,:]
    #3.3 decoder后面部分训练目标 dec_token_matrix[]:,-1,:]
    
    #4.创建mask
    #4.1dec_mask上三角填充-inf的mask
    #4.2enc_pad_mask :(enc矩阵 ==0)
    #4.3 dec_pad_mask:(dec矩阵 ==0)

    #5.创建模型(工具GPU内存大小设计编码和解码器参数和层数)，优化器，损失

    #训练模型并保存



import torch
from torch.utils.data import Dataloader,Dataset
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from transformer_model import Seq2SeqTransformer


#1.构建词典
def build_voacb(token_lists):
    vocab ={'<pd>':0,'<s>':1,'</s>':2}
    idx=3
    for tokens in token_lists:
        for t in tokens:
            if t not in vocab:
                vocab[t] =idx
                idx+= 1
    return vocab

#2.数据集
class MyDataset(Dataset):
    def __init__(self,enc_tokens,dec_tokens,enc_vocab,dec_vocab):
        self.enc_tokens =enc_tokens
        self.dec_tokens =dec_tokens
        self.enc_vocab =enc_vocab
        self.dec_vocab =dec_vocab

    def __len__(self):
        return len(self.enc_tokens)
    
    def __getitem(self,idx):
        enc =[self.enc_vocab[t] for t in self.enc_tokens[idx]]
        dec =[self.dec_vocab[t] for t in self.dec_tokens[idx]]
        return torch.tensor(enc,dtype =torch.long),torch.tensor(dec,dtype=torch.long)
    
#3.collate_fn
def collate_fn(batch):
    enc_batch,dec_batch =zip(*batch)
    enc_batch =pad_sequence(enc_batch,batch_first =True,padding_value =0)
    
    #dec_in：去掉最后一个
    dec_in =[dec[:,:-1] for dec in dec_batch]
    #dec_out:去掉第一个
    dec_out=[dec[:,1:] for dec in dec_batch]

   
    dec_in =pad_sequence(dec_in,batch_first =True,padding_value =0)
    #dec_out:去掉第一个
    dec_out=pad_sequence(dec_out,batch_first=True,padding_value =0)
    return enc_batch,dec_in,dec_out

#4.mask
def generate_square_subsequent_mask(sz):
    mask =torch.triu(torch.ones((sz,sz))*float('-inf'),diagonal =1)
    return mask

#===================主流程================
if __name__ =='__main__':
    #模型数据
    #一批语料： encode：decoder
    #<s></s><pad>
    corpus="人生得意须尽欢，莫使金樽空对月"
    chs=list(corpus)
    
    enc_tokens,dec_tokens=[],[]

    for i in range(1,len(chs)-1) :
        enc=chs[:i]
        dec=['<s>']+chs[i:]+['</s>']
        enc_tokens.append(enc)
        dec_tokens.append(dec)

    #构建词典
    enc_vocab =build_voacb(enc_tokens)
    dec_vocab =build_voacb(dec_tokens)
    inv_dec_vocab ={v: k for k,v in dec_vocab.item()}
    
    #构建数据集和dataloader
    dataset =MyDataset(enc_tokens,dec_tokens,enc_vocab,dec_vocab)
    dataloader =Dataloader(dataset,batch_size =2,shuffle=True,collate_fn=collate_fn)

    #模型参数
    d_model=32
    nhead=4
    num_enc_layers =2
    num_dec_layers=2
    dim_forward =64
    dropout =0.1
    enc_voc_size =len(enc_vocab)
    dec_voc_size=len(dec_vocab)

    divice =torch.device('cuda' if torch.is_available() else 'cpu')
    model =Seq2SeqTransformer(d_model,nhead,num_enc_layers,num_dec_layers)
    optimizer =torch.optim.Adam(model.parameters(),lr =1e-3)
    loss_fn =torch.nn.CrossEntroyLoss(ignore_index=0)

    #训练
    for epoch in range(50):
        model.train()
        total_loss=0
        for enc_batch,dec_in,dec_out in dataloader:
            enc_batch,dec_in,dec_out =enc_batch.to(device),dec_in.to(device),dec_in.to(device),dec_out.to(dec_voc_size)
            tgt_mask =generate_square_subsequent_mask(dec_in.size(1).to(device))
            enc_pad_mask =(enc_batch==0)
            dec_pad_mask =(dec_in ==0)
            logits=model(enc_batch,dec_in,tgt_mask,enc_pad_mask,dec_pad_mask)
            #logits : [B,T,V]  dec_out: [B,T]
            loss =loss_fn(logits.reshape(-1,logits.size(-1)),dec_out.reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss+=loss.item()
        print(f'Epoch {epoch +1},Loss:{total_loss/len(dataloader):.4f}')


    #保存模型
    torch.save(model.state_dict(),'transformer.pth')
    print("模型已保存")







import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import Dataloader
import math

#位置编码矩阵
class PositionalEncoding(nn.Module):

    def __init__(self,emb_size,dropout,maxlen=5000):
        super().__init__()
        #行缩放指数值
        den=torch.exp(torch.arange(0,emb_size,2)*10000/emb_size) #是指数值，所以乘以10000
        #位置编码索引
        pos=torch.arange(0,maxlen).reshape(maxlen,1)
        #编码矩阵
        pos_embedding=torch.zeros(maxlen,emb_size) 
        pos_embedding[:,0::2]=torch.sin(pos*den)  #奇数 列*
        pos_embedding[:,1::2]=torch.cos(pos*den)
        #添加和batch对应的维度(1,5000,emb_size)
        pos_embedding.unsqueeze(-2)    #升维
        #dropout
        self.dropout =nn.Dropout(dropout)
        #注册当前矩阵不参与参数更新
        self.register_buffer('pos_embedding',pos_embedding)
    def forward(self,token_embedding):
        token_len=token_embedding.size(1)   #token长度
        #(1,token_Len,emb_size)
        add_emb=self.pos_embedding [:,token_len,:]+token_embedding
        return self.dropout(add_emb)
    
class Seq2SeqTransformer(nn.Module):

    def __init__(self,d_model,nhead,num_enc_layers,num_dec_layers,dim_forward,
                    dropout,enc_voc_size,dec_voc_size):
        super().__init__()
        #transformer
        self.transformer =nn.Transformer(d_model=d_model,
                                            nhead=nhead,
                                            num_encoder_layers=num_enc_layers,
                                            num_decoder_layers=num_dec_layers,
                                            dim_feedforward =d_model*4,
                                            dropout=dropout,
                                            batch_first=True
                                            )
        #encoder input embedding
        self.enc_emb=nn.Embedding(enc_voc_size,d_model)
        #decoder input embedding
        self.dec_emb=nn.Embedding(dec_voc_size,d_model)
        #predict generate linear
        self.predict =nn.Linear(d_model,dec_voc_size) #token预测基于解码器词典
        #positional encoding
        self.pos_encoding =PositionalEncoding(d_model,dropout)  #不用AI的代码清晰的多

    def forward(self,enc_inp,dec_inp,tgt_mask,enc_pad_mask,dec_pad_mask):
        #multi head attention 之前基于位置编码embeddding生成
        enc_emb =self.pos_encoding(self.enc_emb(enc_inp))
        dec_emb=self.pos_encoding(self.dec_emb(dec_inp))
        #调用transfromer计算
        self.transformer(src=enc_emb,tgt=dec_emb,tgt_mask=tgt_mask,
                            src_key_padding_mask =enc_pad_mask,
                            tgt_key_padding_mask=dec_pad_mask )
        #推理
        return self.predictus(outs)

    #推理环节使用方法
    def encoder(self,enc_inp):
        enc_emb =self.pos_encoding(self.enc_emb(enc_inp))
        return self.transformer.encoder(enc_emb)
 
    def decoder(self,dec_inp,memory,dec_mask):
        dec_emb=self.pos_encoding(self.dec_emb(dec_inp))
                return self.transformer.decoder(dec_emb,memory,dec_mask)


if __name__ =='__main__':

    #模型数据
    #一批语料：encoder：decoder
    #<s></s><pad>
    corpus="人生得意须尽欢，莫使金樽空对月"
    chs=list(corpus)
    
    enc_tokens,dec_tokens=[],[]

    for i in range(1,len(chs)-1) :
        enc=chs[:i]
        dec=['<s>']+chs[i:]+['</s>']
        enc_tokens.append(enc)
        dec_tokens.append(dec)
    print(enc_tokens)
    print(dec_tokens)

