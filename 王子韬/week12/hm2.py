import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoModelForTokenClassification, AutoTokenizer, get_linear_schedule_with_warmup
from datasets import load_dataset
import numpy as np
from tqdm import tqdm
import json
import sys

# 实体类别定义
ENTITIES = ['O', 'movie', 'name', 'game', 'address', 'position',
            'company', 'scene', 'book', 'organization', 'government']


def build_tags():
    tags = ['O']
    for ent in ENTITIES[1:]:
        tags.extend([f'B-{ent.upper()}', f'I-{ent.upper()}'])
    return tags


TAGS = build_tags()
ENT_MAP = {e: i for i, e in enumerate(ENTITIES)}


def setup_ddp(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup():
    dist.destroy_process_group()


def process_data(tokenizer):
    """数据预处理"""
    print("Loading dataset...")
    ds = load_dataset('nlhappy/CLUE-NER')

    # 标注实体标签
    def add_tags(item):
        text_len = len(item['text'])
        tags = [0] * text_len

        for ent in item['ents']:
            idx = ent['indices']
            lbl = ent['label']
            tags[idx[0]] = ENT_MAP[lbl] * 2 - 1  # B-tag
            for i in idx[1:]:
                tags[i] = ENT_MAP[lbl] * 2  # I-tag
        return {'ent_tag': tags}

    ds = ds.map(add_tags)

    # tokenize处理
    def tokenize_fn(batch):
        texts = [list(t) for t in batch['text']]
        enc = tokenizer(texts, truncation=True, add_special_tokens=False,
                        max_length=512, is_split_into_words=True, padding='max_length')
        enc['labels'] = [t + [0] * (512 - len(t)) for t in batch['ent_tag']]
        return enc

    ds = ds.map(tokenize_fn, batched=True)
    ds.set_format('torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'labels'])

    return ds


def train_worker(rank, world_size):
    """训练进程"""
    print(f"Starting worker {rank}/{world_size}")
    setup_ddp(rank, world_size)

    # 初始化
    tokenizer = AutoTokenizer.from_pretrained('google-bert/bert-base-chinese')
    dataset = process_data(tokenizer)

    id2lbl = {i: t for i, t in enumerate(TAGS)}
    lbl2id = {t: i for i, t in enumerate(TAGS)}

    # 模型
    model = AutoModelForTokenClassification.from_pretrained(
        'google-bert/bert-base-chinese',
        num_labels=len(TAGS),
        id2label=id2lbl,
        label2id=lbl2id
    ).to(rank)

    model = DDP(model, device_ids=[rank])

    # dataloader
    sampler = DistributedSampler(dataset['train'], num_replicas=world_size, rank=rank)
    loader = DataLoader(dataset['train'], sampler=sampler, batch_size=16, num_workers=2)

    # optimizer设置 - bert层和分类层不同lr
    bert_params = []
    cls_params = []
    for n, p in model.named_parameters():
        if 'bert' in n:
            bert_params.append(p)
        else:
            cls_params.append(p)

    optim = torch.optim.AdamW([
        {'params': bert_params, 'lr': 1e-5},
        {'params': cls_params, 'lr': 1e-3, 'weight_decay': 0.1}
    ])

    # 训练配置
    epochs = 3
    steps = len(loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optim, num_warmup_steps=100, num_training_steps=steps)
    scaler = torch.cuda.amp.GradScaler()

    # 训练
    model.train()
    for ep in range(epochs):
        sampler.set_epoch(ep)

        pbar = tqdm(loader, desc=f'Epoch {ep + 1}') if rank == 0 else loader

        for batch in pbar:
            batch = {k: v.to(rank) for k, v in batch.items()}
            optim.zero_grad()

            with torch.cuda.amp.autocast():
                loss = model(**batch).loss

            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
            scheduler.step()

            if rank == 0 and hasattr(pbar, 'set_postfix'):
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    # 保存
    if rank == 0:
        save_dir = './ner_model_ddp'
        os.makedirs(save_dir, exist_ok=True)

        torch.save(model.module.state_dict(), f'{save_dir}/pytorch_model.bin')
        model.module.config.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)

        with open(f'{save_dir}/label_mappings.json', 'w') as f:
            json.dump({'id2label': id2lbl, 'label2id': lbl2id, 'tags': TAGS}, f, ensure_ascii=False)

        print(f"Model saved to {save_dir}")

    cleanup()


class NERPredictor:
    """推理器"""

    def __init__(self, model_dir='./ner_model_ddp'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 加载映射
        with open(f'{model_dir}/label_mappings.json', 'r') as f:
            data = json.load(f)
            self.id2lbl = {int(k): v for k, v in data['id2label'].items()}
            self.lbl2id = data['label2id']
            self.tags = data['tags']

        # 加载模型
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForTokenClassification.from_pretrained(
            'google-bert/bert-base-chinese',
            num_labels=len(self.tags),
            id2label=self.id2lbl,
            label2id=self.lbl2id
        )

        ckpt = torch.load(f'{model_dir}/pytorch_model.bin', map_location=self.device)
        self.model.load_state_dict(ckpt)
        self.model.to(self.device).eval()

    def predict(self, text):
        chars = list(text)
        inputs = self.tokenizer(chars, is_split_into_words=True, return_tensors='pt',
                                truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = self.model(**inputs).logits
            preds = torch.argmax(logits, dim=-1)[0].cpu().numpy()

        # 解析实体
        entities = []
        cur_ent = None

        for i, (ch, pid) in enumerate(zip(chars, preds[1:len(chars) + 1])):
            tag = self.id2lbl[pid]

            if tag.startswith('B-'):
                if cur_ent:
                    entities.append(cur_ent)
                cur_ent = {'text': ch, 'type': tag[2:].lower(), 'start': i}
            elif tag.startswith('I-') and cur_ent and tag[2:].lower() == cur_ent['type']:
                cur_ent['text'] += ch
            else:
                if cur_ent:
                    entities.append(cur_ent)
                    cur_ent = None

        if cur_ent:
            entities.append(cur_ent)

        return entities


def run_training():
    n_gpus = torch.cuda.device_count()
    if n_gpus > 0:
        print(f"Training on {n_gpus} GPUs")
        mp.spawn(train_worker, args=(n_gpus,), nprocs=n_gpus, join=True)
    else:
        print("No GPU available!")


def run_inference():
    predictor = NERPredictor()

    test_samples = [
        "浙商银行企业信贷部叶老桂博士则从另一个角度对五道门槛进行了解读。",
        "在北京市海淀区中关村科技园区，华为公司正在举办新产品发布会。",
        "昨天看了《流浪地球2》，导演郭帆的作品真的很震撼。",
        "腾讯游戏推出的《王者荣耀》在中国市场非常受欢迎。"
    ]

    for text in test_samples:
        print(f"\n文本: {text}")
        ents = predictor.predict(text)
        if ents:
            for e in ents:
                print(f"  {e['text']} ({e['type']})")
        else:
            print("  无实体")


if __name__ == "__main__":
    run_training()
    run_inference()