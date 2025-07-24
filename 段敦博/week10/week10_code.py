import torch
import torch.nn as nn
from torch.optim import Adam
from transformers import AutoModelForSequenceClassification
from torch.utils.data import Dataloader
from torch.utils.tensorboard import SummaryWriter
from sklear.model_selection import train_test_split

import pandas as pd
from tqdm import tqdm

device='cuda' if torch.cuda.is_available() else 'cpu'
writer =SummaryWriter()

df =pd.read_excel('/kaggle/input/jd_comment_with_label/jd_comment_data.xlsx')

df.head()

df.columns


filterd =df['评论内容(content)']!="此用户未填写评论内容"
data_df =df[filterd]['评价内容(content)','评分(总分5分)(score)']

data_df.head()

data =data_df.values
data

train,test =train_test_split(data)

print(train.shape)
print(test.shape)

#分词器
tokenize = AutoTokenizer.from_pretrained('hfl/chinese-bert-wwm')


#自定义Dataloader创建方法

def warp_data(batch_data):
    comments,labels=[],[]
    for bdata in batch_data:
        comments.append(bdata[0])
        labels.append(int(bdate[1]-1))  #标签取值[0-4]

    #转换模型输入数据
    input_data=tokenizer(commenys,return_tensors='pt',padding=True,truncation =True,max_length=512)
    labels_data=torch.tensor(labels)

    return input_data,labels_data

train_dl =Dataloader(train,batch_size=20,shuffle=True,collate_fn=warp_data)
test_dl=Dataloader(test,batch_size=20,shuffle=False,collate_fn =warp_data)


#for item in test_dl:
#   printeresting（item)
#   break



#model_1 模型微调 supervied Fine Tuning
#model_2 迁移学习 Transfer Learning 冻结bert
model_1=AutoModelForSequenceClassification.from_pretrained('hfl/chinese-bert-wwm',num_labels=5)
model_2=AutoModelForSequenceClassification.from_pretrained('hfl/chinese-bert-wwm',num_labels=5)

model_2=AutoModelForSequenceClassification.from_pretrained('uer.roberta-base-finetuned-dianping-chinese',
                                                           num_layers=5,
                                                           ignore_missmatch_arguments=True)  #忽略预训练模型中不匹配的层的参数


model_1=model_1.to(device)
model_2=model_2.to(device)
model_2.bert.trainable=False    #TF应用

#bert.embedding.xxx
#bert.xxx

#with torch.no_grad():
#    out=model_2.bert(**model_input)
#final= model_2.classifier(out)


#loss,optim
loss_fn1 =nn.CrossEntropyLoss()
optim1=Adam(model_1.parameters(),lr=1e-4)

loss_fn2 =nn.CrossEntropyLoss()
optim2=Adam(model_2.parameters(),lr=1e-4)

model1_train_loss_cnt=0

for epoch in range(3):
    pbar =tqdm(train_dl)
    for input_data,labels_data in pbar:
        datas ={k:v to(device) for k,v in input_data.items()}
        labels =labels_data.to(device)

        result =model_1(**datas)
        loss =loss_fn1(result.logits,labels)

        pbar.set_description(f'epoch:{epoch} train_loss:{loss.item():.4f}')

        writer.add_scalar('Fine Tuning Train Loss',loss,model1_train_loss_cnt)
        model1_train_loss_cnt+=1

        loss.backward()
        optim1.step()

        model_1.zero_grad()

torch.save(model_1.state_dict(),'model_1.pt')


model_1.eval()
model_2.eval()
pbar=tqdm(test_dl)

correct1,corrct2=0,0

for input_data,labels_data in pbar:
    datas={k:v.to(device) for k,v in input_data.items()}
    labels=labels_data.to(device)

    with torch.mp_grad():
        result1=model_1(**datas)
        result2=model_2(**datas)

    predict1=torch.argmax(result1.logits,dim=-1)
    predict2=torch.argmax(result2.logits,dim=-1)

    correct1+=(predict1 ==labels).sum()
    correct2=(predict2==labels).sum()
    
