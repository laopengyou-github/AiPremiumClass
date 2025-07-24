import torch
import torch.nn as nn
import torch.nn.functional as F

def get_batch(split):
    """
    获取训练或验证批次数据
    
    Args:
        split (str): 'train' 或 'val'，指定获取训练集还是验证集
    
    Returns:
        tuple: (x, y) 输入序列和目标序列的批次数据
    """
    # 根据split参数选择对应的数据集
    data = train_data if split == 'train' else val_data

    # 随机生成batch_size个起始位置索引，确保每个序列都能完整截取block_size个字符
    # torch.randint(len(data) - block_size, (batch_size,)) 生成范围在[0, len(data)-block_size)的随机索引
    ix = torch.randint(len(data) - block_size, (batch_size,)) 
    
    # 构建输入序列x：从每个起始位置截取block_size个字符
    x = torch.stack([data[i:i+block_size] for i in ix])
    # 构建目标序列y：从每个起始位置+1截取block_size个字符（即x的右移一位）
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])

    return x.to(device), y.to(device)

class Head(nn.Module):
    """
    单头自注意力机制 (Single Head Self-Attention)
    实现Transformer中的核心注意力计算
    """
    def __init__(self, n_embd):
        """
        初始化注意力头
        
        Args:
            n_embd (int): 嵌入维度，即每个token的特征维度
        """
        super().__init__()
        # 定义三个线性变换层，用于生成Query、Key、Value矩阵
        # bias=False 表示不使用偏置项，这是注意力机制的标准做法
        self.key = nn.Linear(n_embd, n_embd, bias=False)    # 生成Key矩阵
        self.query = nn.Linear(n_embd, n_embd, bias=False)  # 生成Query矩阵
        self.value = nn.Linear(n_embd, n_embd, bias=False)  # 生成Value矩阵

    def forward(self, input_x):
        """
        前向传播：计算自注意力
        
        Args:
            input_x (torch.Tensor): 输入张量，形状为 (B, T, C)
                                   B: batch_size, T: 序列长度, C: 嵌入维度
        
        Returns:
            torch.Tensor: 注意力输出，形状为 (B, T, C)
        """
        B, T, C = input_x.shape  # 获取批次大小、序列长度、嵌入维度

        # 通过线性变换生成Q、K、V矩阵
        k = self.key(input_x)    # (B, T, C)
        q = self.query(input_x)  # (B, T, C)
        v = self.value(input_x)  # (B, T, C)

        # 计算注意力权重：Q @ K^T / sqrt(C)
        # q @ k.transpose(-2,-1) 计算Q和K的相似度矩阵
        # * C ** -0.5 进行缩放，防止梯度消失
        wei = q @ k.transpose(-2,-1) * C ** -0.5  # (B, T, T)
        
        # 创建下三角掩码矩阵，实现因果注意力（只能看到当前位置及之前的信息）
        T = wei.shape[-1]
        tril = torch.tril(torch.ones(T,T, device=device))  # 下三角矩阵，形状为(T, T)
        # 将上三角部分（未来信息）的权重设为负无穷，经过softmax后变为0
        wei = wei.masked_fill(tril == 0, float('-inf'))
        # 对注意力权重进行softmax归一化
        wei = wei.softmax(dim=-1)  # (B, T, T)

        # 计算注意力输出：权重矩阵与Value矩阵相乘
        out = wei @ v  # (B, T, C)
        return out


class BingramLanguageModel(nn.Module):
    """
    二元语言模型 (Bigram Language Model)
    基于Transformer架构的简化语言模型，用于文本生成
    """
    
    def __init__(self, block_size, vocab_size, n_embd):
        """
        初始化语言模型
        
        Args:
            block_size (int): 上下文窗口大小，即模型能看到的字符数量
            vocab_size (int): 词汇表大小，即所有可能字符的数量
            n_embd (int): 嵌入维度，即每个字符的特征维度
        """
        super().__init__()
        # Token嵌入层：将字符索引转换为向量表示
        # 每个token都直接从Embedding中查询对应的logits值以进行下一个token的推理
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        
        # 位置嵌入层：为序列中的每个位置提供位置信息
        # 由于Transformer没有循环结构，需要显式的位置编码
        self.position_embedding_table = nn.Embedding(block_size, n_embd)

        # 单头自注意力层
        self.sa_head = Head(n_embd)
        
        # 语言模型头：将隐藏状态映射到词汇表大小的logits
        self.lm_head = nn.Linear(n_embd, vocab_size)
        
    def forward(self, idx, targets=None):
        """
        前向传播
        
        Args:
            idx (torch.Tensor): 输入字符索引，形状为 (B, T)
            targets (torch.Tensor, optional): 目标字符索引，用于计算损失
        
        Returns:
            tuple: (logits, loss) 预测logits和损失值
        """
        B, T = idx.shape  # 获取批次大小和序列长度
        
        # idx值和targets值都是整型张量 (B,T)
        # 将字符索引转换为嵌入向量
        tok_emb = self.token_embedding_table(idx)  # (B,T,C)
        
        # 生成位置嵌入：为序列中的每个位置生成位置编码
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        
        # 将token嵌入和位置嵌入相加得到最终的嵌入表示
        x = tok_emb + pos_emb  # (B,T,C)
        
        # 通过自注意力层处理序列
        x = self.sa_head(x)  # (B,T,C)
        
        # 通过语言模型头得到每个位置的logits
        logits = self.lm_head(x)  # (B,T,vocab_size)
        
        if targets is None:
            # 推理模式：不计算损失
            loss = None
        else:
            # 训练模式：计算交叉熵损失
            B, T, C = logits.shape
            # 将logits重塑为 (B*T, C) 以便计算损失
            logits = logits.view(B*T, C)
            # 将targets重塑为 (B*T,) 
            targets = targets.view(-1)
            # 计算交叉熵损失
            loss = F.cross_entropy(logits, targets)
        
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        """
        文本生成函数
        
        Args:
            idx (torch.Tensor): 起始字符索引，形状为 (B, T)
            max_new_tokens (int): 要生成的新token数量
        
        Returns:
            torch.Tensor: 生成的完整序列
        """
        # idx指当前语料集(B,T)中的索引
        for _ in range(max_new_tokens):
            # 限定索引列的取值范围，只使用最后block_size个token作为上下文
            # 这是为了处理序列长度限制
            idx_cond = idx[:, -block_size:]
            
            # 前向推理，得到logits
            logits, loss = self(idx_cond)
            
            # 只提取最后一个时间步的结果，用于预测下一个token
            logits = logits[:, -1, :]  # (B,C)
            
            # 通过softmax将logits转换为概率分布
            probs = F.softmax(logits, dim=-1)  # (B,C)
            
            # 根据概率分布随机采样下一个token
            idx_next = torch.multinomial(probs, num_samples=1)  # (B,1)
            
            # 把采样的索引追加在当前解码序列末尾
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx

if __name__ == '__main__':

    # ==================== 模型超参数设置 ====================
    block_size = 8      # 上下文窗口大小：模型能看到的字符数量
    batch_size = 32     # 批次大小：每次训练处理的样本数量
    max_iter = 10000    # 最大训练迭代次数
    learn_rate = 1e-3   # 学习率
    device = 'cuda' if torch.cuda.is_available() else 'cpu'  # 设备选择
    n_embd = 64         # 嵌入维度：每个字符的特征维度
    eval_interval = 500 # 评估间隔：每训练多少步评估一次
    eval_iters = 200    # 评估迭代次数：评估时计算多少批次的平均损失

    # ==================== 数据加载和预处理 ====================
    # 读取训练文本文件
    with open('homework/week16/HLM.txt', 'r', encoding='utf-8') as f:
        text = f.read()

    # 构建词汇表和编码器/解码器
    # 获取文本中所有唯一字符并排序
    chars = sorted(list(set(text)))
    vocab_size = len(chars)  # 词汇表大小
    
    # 创建字符到索引的映射字典
    stoi = {ch:i for i,ch in enumerate(chars)}  # str_to_index
    # 创建索引到字符的映射字典
    itos = {i:ch for i,ch in enumerate(chars)}  # index_to_str

    # 定义编码和解码函数
    encode = lambda s: [stoi[c] for c in s]  # 将字符串转换为索引列表
    decode = lambda l: ''.join([itos[i] for i in l])  # 将索引列表转换为字符串

    # 将整个文本转换为token索引序列
    data = torch.tensor(encode(text), dtype=torch.long)

    # 拆分数据集：90%用于训练，10%用于验证
    n = int(len(data) * .9)
    train_data = data[:n]  # 训练集
    val_data = data[n:]    # 验证集

    # ==================== 模型初始化和训练准备 ====================
    # 创建模型实例
    model = BingramLanguageModel(block_size, vocab_size, n_embd)
    model.to(device)  # 将模型移动到指定设备（GPU/CPU）
    
    # 创建优化器：使用AdamW优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=learn_rate)

    @torch.no_grad()  # 禁用梯度计算，节省内存
    def estimate_loss():
        """
        评估函数：计算训练集和验证集的平均损失
        
        Returns:
            dict: 包含训练集和验证集损失值的字典
        """
        out = {}
        model.eval()  # 设置为评估模式，禁用dropout等
        for split in ['train', 'val']:
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                X, Y = get_batch(split)  # 获取批次数据
                logits, loss = model(X, Y)  # 前向传播
                losses[k] = loss.item()  # 记录损失值
            out[split] = losses.mean()  # 计算平均损失
        model.train()  # 恢复训练模式
        return out

    # ==================== 模型训练循环 ====================
    for iter in range(max_iter):

        # 定期评估模型性能
        if iter % eval_interval == 0:
            losses = estimate_loss()
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        # 获取训练批次数据
        xb, yb = get_batch('train')

        # 前向传播：计算预测和损失
        logits, loss = model(xb, yb)
        
        # 反向传播：计算梯度
        optimizer.zero_grad(set_to_none=True)  # 清零梯度
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数

    # ==================== 模型推理和文本生成 ====================
    # 创建起始token：全零张量，形状为(1,1)
    idx = torch.zeros((1,1), dtype=torch.long, device=device)
    
    # 生成500个新token并打印结果
    generated_text = decode(model.generate(idx, max_new_tokens=500)[0].tolist())
    print(generated_text) 