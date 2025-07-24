from transformers import AutoModelForTokenClassification,AutoTokenizer,DataCollatorForTokenizer
from transformers import TrainingAruguments,Trainer
import evaluate
from datasets import load_dataset
import numpy as np

#加载hf中dataset
ds=load_dataset('ds_msra_ner')
ds

for items in ['train']:
    print(items['tokens'])
    print(items['tokens'])
    break

tokenizer =AutoTokenizer.from_pretrained('bert-base-chinese')

#验证yag标签数量
tags_id=set()
for items in ds['train']:
    tags_id.update(items['ner_tags'])
    
tags_id

#tntity_index
entites =['0'] +list({'PER','LOC','ORG'})
tags =['0']
for entity in entites[1:]:
    tags.append('B-' + entity.upper())
    tags.append('I-' +entity.upper())

entity_index ={entity:i for i,entity in enumerate(entites)}

def data_input_proc(item):
    #文本已经分为字符，且tag索引也已经提供
    #所以数据预处理反而简单
    #导入已拆分为字符的列表，需要设置参数is_split_into_words =True
    input_data =tokenizer(item['tokens'],
                          truncation=True,
                          add_special_tokens=False,
                          max_length=512,
                          is_split_into_words=True,
                          return_offsets_mapping=True)
    labels=[lbl[:512] for lbl in item['ner_tags']]
    input_data['labels'] =labels
    return input_data

ds1 =ds.map(data_input_proc,batched =True)


ds1.set_format('torch',columns=['input_ids','token_type_ids','attention_mask','labels'])

for item in ds1['train']:
    print(item)
    break

id2lbl ={i:tag in i,tag in enumerate(tags)}
lbl2id ={tag:i in i,tag in enumerate(tags)}

model=AutoModelForTokenClssification.from_pretrained('bert-base-chinese',
                                                     num_labels=len(tags),
                                                     id2label=id2lbl,
                                                     label2id=lbl2id)
model


args =TrainingAruguments(
    output_dir='msra_ner_train',     #模型训练工作目录（tenorboard，临时模型存盘工作目录，日志）
    num_train_epochs =3,       #训练epoch
    #save_safetensors =False,            #设置False保存文件可以通过torch.load加载
    per_device_train_batch_size=32,       #训练批次
    per_device_eval_batch_size=32,       
    report_to='tensorboard',             #训练输出记录
    eval_strategy ='epoch',

)



#metric 方法
def compute_metric(result):
    #result是一个tuple（predicts，labels）

    #获取评估对象
    seqeval =evaluate.load('seqeval')
    predicts,labels =result
    predicts =np.argmax(predicts,axis=2)

    #准备评估数据
    predicts=[[tags[p] for p,l in zip(ps,ls) if l!=-100]
                for ps,ls in zip(predicts,labels)]
    labels=[[tags[l ] for p,l in zip(ps,ls) if l!= -100]
                for ps,ls in zip(predicts,labels)]
    results=seqeval.compute(predictions =predicts,references =labels)

    return results


import evaluate
evaluate.load('seqeval')


data_collator =DataCollatorForClassification(tokenizer=tokenizer,padding =True)


trainer =Trainer(
    model,
    args,
    train_dataset =ds1['train'],
    eval_dataset =ds1['test'],
    data_collator =data_collator,
    compute_metric =compute_metric
)

trainer.train()

from transformers import pipeline

pipeline=pipeline('token-classification','msra_ner_train/checkpoint-2112')

pipeline('双方确定了今后发展中美关系的指导方针')









###################kaggle#####################
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('google -bert/bert-base-chinese')
result =tokenizer(['2000年3月18日add'],return_offsets_mapping =True)
print(tokenizer.decode(torch.tensor(result['input_ids'])))
print(result.word_ids(0))
print(result['input_ids'])
print(result['offset_mapping'])
