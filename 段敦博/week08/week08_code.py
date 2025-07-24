import torch
import torch.nn as nn

#编码器
class Encoder(nn.Moduel):
    def __init__(self,input_dim,emb_dim,hidden_dim,num_layers=2,state_type='concat')
        super(Encoder,self).__init__()
        #定义嵌入层
        self.embedding=nn.Embedding(input_dim,emb_dim)
        #定义GRU层
        self.rnn=nn.GRU(emb_dim,hidden_dim,num_layers=num_layers,
                        batch_first=True,bidirectional=True)
        #自定义返回state类型(concat,add)
        self.stete_type=state_type

    def forward(self,token_seq):
        #token_seq:[batch_size,emb_dim]
        #embedded:[batch_size,seq_len,emb_dim]
        embedded =self.embedding(token_seq)
        #outputs:[batch_size,seq_len,emb_dim]
        #hidden:[2,batch_size,hidden_dim]
        outputs,hidden =self.rnn(embedded)
        #返回,Encoder最后一个时间步的隐藏状态(拼接)
        #return outputs[:,-1,:]
        
        if self.state_type=='concat':
            #最后一个时间步的隐藏状态
            hidden =outputs[:,-1,:]
        elif self.state_type =='add':
            #返回最后一个时间步的隐藏状态
            hidden =torch.sum(hidden,dim=0)
            outputs=outputs[...,:250]+outputs[...,250:]
        else:
            raise ValueError("state_type must be 'concat' or 'add' ")
        return hidden,outputs

class Attention(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self,enc_output,dec_output):
        #a_t =h_t@h_s
        a_t  =torch.bmm(enc_output,dec_output.premute(0,2,1))
        #1.计算 结合解码token和编码token，管理的权重
        a_t =torch.softmax(a_t,dim=1)
        #2.计算 关联权重和编码token 贡献值
        c_t =torch.bmm(a_t.permute(0,2,1),enc_output)
        return c_t
    
#解码器
class Decoder(nn.Module):
    def __init__(self,input_dim,emb_dim,hidden_dim,dropout=0.5,state_type='concat'):
        super(Decoder,self).__init__()
        if state_type =='concat':
            hidden_dim =hidden_dim*2
        #定义嵌入层
        self.embedding=nn.Embedding(input_dim,emb_dim)
        #定义GRU层
        self.rnn=nn.GRU(emb_dim,hidden_dim,batch_first=True)
        #定义线性层
        self.fc=nn.Linear(hidden_dim,input_dim) #解码词典中词汇概率
        #attention层
        self.attention=Attention()
        #attention结果转换线性层
        self.attention_fc=nn.linear(hidden_dim*2,hidden_dim)
        #dropout层
        self.dropout=nn.Dropout(dropout)

    def forward(self,hidden_state,enc_output):
        #token_seq:[batch_size,emb_dim]
        #embedded:[batch_size,seq_len,emb_dim]
        embedded =self.embedding(token_seq)
        #outputs:[batch_size,seq_len,hidden_dim]
        #hidden:[1,batch_size,hidden_dim]
        dec_output,hidden =self.rnn(embedded,hidden_state.unsqueeze(0))

        #attention运算
        c_t=self.attention(enc_output,dec_output)
        #[attention,dec_output]
        cat_output=torch.cat((c_t,dec_output),dim=-1)
        #dropout
        out=self.dropout(out)
        #out:[batch_size,seq_len,hidden_dim*2]
        logits=self.fc(out)
        return logits,hidden
    
class Seq2Seq2(nn.Module):

    def __init__(self,
                 enc_emb_size,
                 dec_emb_size,
                 emb_dim,
                 hidden_size,
                 dropout=0.5,
                 state_type='concat'
                 ):
        super().__init__()

        #encoder
        self.encoder =Encoder(enc_emb_size,emb_dim,hidden_dim,state_type=state_type,)
        #decoder
        self.decoder=Decoder(dec_emb_size,emb_dim,hidden_dim,dropout,state_type=state_type)
    
    def forward(self,enc_input,dec_input):
        #encoder last hidden state
        encoder_state,outputs =self.encoder(enc_input)
        output,hidden =self.decoder(dec_input,encoder_state,outputs)




if __name__ =='__main__':
    #测试Encoder
    input_dim=200
    emb_dim=256
    hidden_dim=256
    dropout=0.5
    batch_size=5
    seq_len=10

    #encoder=Encoder(input_dim,emb_dim,hidden_dim,dropout)
    #token_seq =torch.randint(0,input_dim,(batch_size,seq_len)
    #hidden_state=encoder(token_Seq)    #Encoder输出（最后时间步状态）
    ##print(hidden_state.shape)   #应该是[batch_size,hidden_dim]
    
    #测试Decoder
    decoder=Decoder(input_dim,emb_dim,hidden_dim,dropout)
    token_seq =torch.randint(0,input_dim,(batch_size,seq_len))
    logits=decoder(token_seq,hidden_dim)   #Decoder输出






import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataloader
from torch.nn.utils.rnn import pad_sequence

def read_data(in_file,out_file):
    '''
    读取训练数据返回设计集合
    '''
    enc_data,dec_data=[],[]
    in_=open(in_file)
    out_=open(out_file)

    for enc,dec in zip(in_,out_):
        #分词
        enc_tks=enc.split()
        dec_tks=dec.split()
        #保存
        enc_data.append(enc_tks)
        dec_data.append(dec_tks)

    #断言
    assert len(enc_data) == len(dec_data),'编码数据与解码数据长度不一致！'

    return enc_data,dec_data

def get_proc(vocab):

    #嵌套函数定义
    #外部函数变量生命周期会延续到内部函数调用结束（闭包）

    def batch_proc(data):
        '''
        批次数据处理并返回
        '''
        enc_ids,dec_ids,labels=[],[]
        for enc,dec in data:
            #token ->token index 首位添加起始和结束token
            enc_idx =[vocab['<s>']]+[vocab[tk] for tk in enc] + [vocab['</s>']]
            dec_idx=[vocab['<s>']]+[vocab[tk] for tk in dec] + [vocab['</s>']]
        #encoder_input
        enc_ids.append(torch.tensor(enc_idx))
        #decoder_input
        dec_ids.append(torch.tensor(dec_idx[:-1]))
        #label
        labels.append(torch.tensor(dec_idx[1:]))
    
    #数据转换张量[batch,max_token_len]
    #用批次中最长token序列构建张量
    enc_input=pad_sequence(enc_ids,batch_first=True)
    dec_input =pad_sequence(dec_ids,batch_first=True)
    targets=pad_sequence(labels,batch_first=True)

    #返回数据都是模型训练和推理的需要
    return enc_input,dec_input ,targets

class Vocabulary:

    def __init__(self,vocab):
        self.vocab=vocab

    @classmethod
    def from_file(cls,vocab_file):
        with open(vocab_file,encoding='utf-8') as f:
            vocab=f.read().split('\n')
            vocab=['<pad>'] +[tk for tk in vocab if tk!='']
            #<s>和</s>作为字符串的起始和结束token
            return cls({tk:i for i,tk in enumerate(vocab)})


if __name__ == '__main__':

    #加载词典
    vocab_file ='couplet/vocab'
    vocab=Vocabulary.from_file(vocab_file)

    #训练数据
    enc_train_file='couplet/train/in.txt'
    dec_train_file='couplet/train/out.txt'

    enc_data,dec_data=read_data(enc_train_file,dec_train_file)

    print('enc length':,len(enc_data))
    print('dec length':,len(dec_data))

    print('词汇数量':,len(vocab.vocab))

    #编码+解码(训练样本)
    dataset =list(zip(enc_data,dec_data))
   
    #Dataloader
    dataloader=Dataloader(
        dataset,
        batch_size=2,
        shuffle=True,
        colla_fn=get_proc(vocab.vocab)  #callback
    )


    #数据缓存
    import json

    #数据整体json数据集（json）
    with open('encoder.json','w',encoding='utf-8') as f:
        json.dump(enc_data,f)


    #测试
    #数据每行都是json数据（jsonl)
    #with open('encoders.json','w',encoding='utf-8') as f:
     #   for enc in enc_data:
    #        str_json=json.dumps(enc)
    #        f.write(str_json +'\n')


'''
1.加载训练好模型和词典
2.解码推理流程
    -用户输入通过vocab转换token_index
    -token_index通过encoder获取encoder last hidden_state
    -准备decoder 输入第一个token_index:[['BOS]] shape:[1,1]
    -循环decoder
        -decoder输入:[['BOS]],hidden_state
        -decoder输出:output,hidden_state output shape:[1,1,dec_voca_size]
        -计算argmax'，得到下一个token_index
        -decoder的下一个输入=token_index
        -收集每次token_index[解码集合]
    -输出解码结果'''

import torch
import pickle
import random
from process import read_data,get_proc,Vocabulary
from EncoderDecoderAttenModel import Seq2Seq2

if __name__=='__main__':
    
    enc_test_file='couplet/test/in.txt'
    dec_test_file='couplet/test/out.txt'
    
    enc_data_file='couplet/test/in.txt'
    dec_data_file='couplet/test/out.txt'

    enc_data,dec_data =read_data(enc_data_file,dec_data_file)

    #加载训练好的模型和词典
    state_dict=torch.load('seq2seqadd_satate.bin')
    vocab_file='couplet/vocabs'
    vocab=Vocabulary.from_file(vocab_file)

    model =Seq2Seq2(
        enc_emb_size=len(vocab.vocab),
        dec_emb_size=len(vocab.vocab),
        emb_dim=200,
        hidden_size=250,
        dropout=0.5,
        state_type='add',
    )
    model.load_state_dict(state_dict)

    #创建解码器反向字典
    dvoc_inv={v:k for k,v in vocab.vocab.items()}
    
    #随机选取测试样本
    rnd_idx=random.randint(0,len(enc_data))
    enc_input=enc_data[rnd_idx]
    dec_output=dec_data[rnd_idx]
    
    enc_idx=torch.tensor([vocab.vocab[tk] for tk in enc_input])

    print(enc_idx.shape)

    #推理
    #最大解码长度
    max_dec_len=len(enc_input)

    model.eval()
    with torch.no_grad():
        #编码器输出
        #hidden_state=model.encoder(enc_idx)
        hidden_state,enc_outputs=model.encoder(enc_idx)

        #解码器输入 shaoe[1,1]
        dec_input =torch.tensor([[vocab.vocab['<s>']]])

        #循环 decoder
        dec_tokens=[]
        while True:
            if len(dec_tokens) >=max_dec_len:
                break
            #解码器
            #logits：[1,1,dec_voc_size]
            #logits,hidden_state =model.decoder(dec_input,hidden_state)
            logits,hidden_state =model.decoder(dec_input,hidden_state,enc_outputs)

            #下个token index
            next_token =torch.argmax(logits,dim=-1)

            if dvoc_inv [next_token.squeeze().item()] =='</s>':
                break
            #收集每次token_index [解码集合]
            dec_tokens.append(next_token.squeeze().item())
            #decoder的下一个输入 =token_index
            dec_input =next_token
            hidden_state =hidden_state.view(1,-1)

    #输出解码结果
    print(f'上联：',''.join(enc_input))
    print('模型预测下联：','').join([dvoc_inv[tk] for tk in dec_tokens])
    print('真实下联：',''.join(dec_output))




import pickle
import torch
import json
from torch.utils.data import Dataloader
from process import get_proc,Vocabulary
from EncoderDecoderAttenModel import Seq2Seq2
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

if __name__ =='__main__':

    writer=SummaryWriter()
    train_loss_cnt=0
    device=torch.device('cuda')

    #加载训练数据
    vocab_file='couplet/vocabs'
    vocab=Vocabulary.from_file(vocab_file)

    with open('encoder.json') as f:
        enc_data=json.load(f)
    with open('decoder.json') as f:
        dec_data=json.load(f)
    
    ds=list(zio(enc_data,dec_data))
    dl=Dataloader(ds,batch_size=256,shuffle=True,collate_fn=get_proc(vocab.vocab))

    #构建训练模型
    #模型构建
    model=Seq2Seq2(
        enc_emb_size=len(vocab.vocab),
        dec_emb_size=len(vocab.vocab),
        emb_dim=200,
        hidden_size=250,
        dropout=0.5,
        #state_type ='add'
    )
    model.to(device)

    #优化器，损失
    optimizer =optim.Adam(model.parameters(),lr=1e-3,weight_decay=1e-5)
    criterion=nn.CrossEntroyLoss()

    #训练
    for epoch in range(20):
        model.train()
        tpbar=tqdm(dl)
        for enc_input,dec_input,targets in tpbar:
            enc_input =enc_input.to(device)
            dec_input=dec_input.to(device)
            targets =targets.to(device)

            #前向传播
            logits,_ =model(enc_input.dec_input)

            #计算损失
            #CrossEntroyLoss 需要将logits和targets展平
            #logits:[batch_size,seq_len,vocab_size]
            #展平为[batch_size*seq_len,vocab_size] 和[batch_size*seq_len]

            #反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            tpbar.set_description(f'Epoch {epoch +1},Loss:{loss.item():.4f}')
            writer.add_scalar=('Loss/train',loss.item(),train_loss_cnt)
            train_loss_cnt+=1

    torch.save(model.state_dict(),'seq2seq_state.bin')
