import pickle
import torch
import json
from torch.utils.data import DataLoader
from process import get_proc
from EncoderDecoderAttenModel import Seq2Seq
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


if __name__ == '__main__':
    writer = SummaryWriter(log_dir='runs/couplet_lstm')
    device = torch.device('cuda')

    # 加载训练数据
    with open('vocab.bin','rb') as f:
        evoc,dvoc = pickle.load(f)

    with open('encoder.json') as f:
        enc_data = json.load(f)
    with open('decoder.json') as f:
        dec_data = json.load(f)

    ds = list(zip(enc_data,dec_data))
    dl = DataLoader(ds, batch_size=256, shuffle=True, collate_fn=get_proc(evoc, dvoc))

    # 构建训练模型
    # 模型构建
    model = Seq2Seq(
        enc_emb_size=len(evoc),
        dec_emb_size=len(dvoc),
        emb_dim=100,
        hidden_size=120,
        dropout=0.5,
    )
    model.to(device)

    # 优化器、损失
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    # 训练
    for epoch in range(40):
        model.train()
        tpbar = tqdm(dl)
        for enc_input, dec_input, targets in tpbar:
            enc_input = enc_input.to(device)
            dec_input = dec_input.to(device)
            targets = targets.to(device)

            # 前向传播 
            logits, _ = model(enc_input, dec_input)

            # 计算损失
            # CrossEntropyLoss需要将logits和targets展平
            # logits: [batch_size, seq_len, vocab_size] vocab_size 表示词汇表的大小
            # targets: [batch_size, seq_len]
            # 展平为 [batch_size * seq_len, vocab_size] 和 [batch_size * seq_len]
            #-1 表示自动计算该维度的大小 logits.size(-1) 表示取 logits 的最后一维（即词汇表大小 vocab_size）。
            # targets.view(-1) 将 targets 展平为一维张量，形状为 [batch_size * seq_len]。 
            loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            tpbar.set_description(f'Epoch {epoch+1}, Loss: {loss.item():.4f}')
            writer.add_scalars('Loss', {'train': loss.item()}, epoch)

    torch.save(model.state_dict(), 'seq2seq_state.bin')