from homework_01 import *

# 加载模型进行推理
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

model.load_state_dict(torch.load('transformer_couplet_model.pth'))  # 加载模型权重
model.eval()  # 设置为评估模式

def text_to_indices(sentences, vocab):
    indices_list = []
    for sent in sentences:
        indices = []
        for char in sent:
            indices.append(vocab.get(char, vocab['<unk>']))
        indices_list.append(torch.tensor(indices, dtype=torch.long))
    return indices_list

def indices_to_text(indices, vocab):
    text_list = []
    for idx in indices:
        if idx in vocab.values():  # 检查索引是否存在于词汇表中
            text_list.append(list(vocab.keys())[list(vocab.values()).index(idx)])
        else:
            text_list.append('<unk>')  # 索引不存在时使用<unk>替代
    return ''.join(text_list)

# 示例输入
input_sentence = "春风入喜财入户"
input_indices = text_to_indices([input_sentence], enc_vocab)
input_tensor = pad_sequence(input_indices, batch_first=True, padding_value=0).to(device)

# 编码
memory = model.encode(input_tensor)

# 解码初始化
dec_input = torch.tensor([[dec_vocab['<sos>']]]).to(device)  # 解码器初始输入为<sos>

# 循环生成下联
generated_indices = []
with torch.no_grad():
    for _ in range(100):  # 最大生成长度
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(dec_input.size(1)).to(device)
        output = model.decode(dec_input, memory, tgt_mask)
        output = model.predict(output)
        next_token = output.argmax(2).squeeze(0)[-1].item()  # 获取最后一个词的预测结果
        generated_indices.append(next_token)
        dec_input = torch.cat([dec_input, torch.tensor([[next_token]]).to(device)], dim=1)
        if next_token == dec_vocab['<eos>']:  # 遇到<eos>停止生成
            break

# 转换生成的索引为文本
generated_sentence = indices_to_text(generated_indices, dec_vocab)
print("上联:", input_sentence)
print("下联:", generated_sentence)