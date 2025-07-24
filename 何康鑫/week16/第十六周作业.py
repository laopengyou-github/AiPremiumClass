from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain import hub
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.tools import BaseTool
import pdfplumber

# 自定义PDF提取工具
class PDFExtractor(BaseTool):
    name = "pdf_extractor"
    description = "Extract text from a specified PDF file."

    def __init__(self, pdf_path):
        self.pdf_path = pdf_path

    def _run(self, query):
        # 简化实现：直接返回整个PDF文本
        with pdfplumber.open(self.pdf_path) as pdf:
            return "\n".join(page.extract_text() for page in pdf.pages)

# 初始化工具
pdf_tool = PDFExtractor(pdf_path="example.pdf")
search_tool = TavilySearchResults(max_results=2)

# 获取提示模板
prompt = hub.pull("hwchase17/openai-functions-agent")

# 创建Agent
agent = create_tool_calling_agent(
    llm=ChatOpenAI(model="gpt-3.5-turbo"),
    tools=[pdf_tool, search_tool],
    prompt=prompt
)

# 执行Agent
executor = AgentExecutor(agent=agent, tools=[pdf_tool, search_tool], verbose=True)

response = executor.invoke({
    "input": "请结合《Python编程导论》第3章内容，解释递归函数的工作原理，并举例说明"
})

print(response["answer"])
import torch
from collections import defaultdict

# 加载中文语料
with open("chinese_text.txt", "r", encoding="utf-8") as f:
    text = f.read()

# 构建字符级词汇表
chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

# 数据预处理
encoded = torch.tensor([stoi[ch] for ch in text], dtype=torch.long)
block_size = 64  # 序列长度

# 创建数据加载器
def data_loader():
    while True:
        idx = torch.randint(0, len(encoded) - block_size, (32,))
        yield encoded[idx:idx+block_size], encoded[idx+1:idx+block_size+1]
      import torch.nn as nn
import torch.nn.functional as F

class Head(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.key = nn.Linear(n_embd, n_embd, bias=False)
        self.query = nn.Linear(n_embd, n_embd, bias=False)
        self.value = nn.Linear(n_embd, n_embd, bias=False)
    
    def forward(self, x):
        B, T, C = x.size()
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * (C ** -0.5)
        wei = wei.softmax(dim=-1)
        return wei @ self.value(x)

class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, n_embd):
        super().__init__()
        self.heads = nn.ModuleList([Head(n_embd) for _ in range(n_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
    
    def forward(self, x):
        return self.proj(torch.cat([h(x) for h in self.heads], dim=-1))

class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd)
        )
    
    def forward(self, x):
        return self.net(x)

class GPT(nn.Module):
    def __init__(self, vocab_size, n_embd=128, n_head=4, n_layers=6):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Parameter(torch.zeros(1, 1024, n_embd))
        self.layers = nn.ModuleList([
            nn.Sequential(
                MultiHeadAttention(n_head, n_embd),
                nn.LayerNorm(n_embd),
                FeedForward(n_embd),
                nn.LayerNorm(n_embd)
            ) for _ in range(n_layer)
        ])
        self.lm_head = nn.Linear(n_embd, vocab_size)
    
    def forward(self, idx):
        B, T = idx.size()
        pos = torch.arange(T, device=idx.device).unsqueeze(0).expand(B, -1)
        x = self.token_emb(idx) + self.pos_emb[:, :T]
        
        for layer in self.layers:
            x = layer(x)
        
        return self.lm_head(x)

    def generate(self, idx, max_new_tokens=500):
        for _ in range(max_new_tokens):
            # 只取最后block_size长度的序列
            idx_cond = idx[:, -block_size:]
            logits = self(idx_cond)
            logits = logits[:, -1, :] / 0.7  # 温度采样
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
      import torch.optim as optim
from data import data_loader
from model import GPT

# 参数配置
batch_size = 32
block_size = 64
n_layers = 6
n_head = 4
n_embd = 128
learning_rate = 3e-4
num_epochs = 5

# 初始化模型和优化器
model = GPT(vocab_size=vocab_size, n_embd=n_embd, n_head=n_head, n_layer=n_layer)
optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

# 训练循环
model.train()
for epoch in range(num_epochs):
    data_gen = data_loader()
    total_loss = 0
    for batch_idx, (data, target) in enumerate(data_gen):
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output.view(-1, vocab_size), target.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(data_gen):.4f}")
from model import GPT

# 加载预训练模型
model = GPT(vocab_size=vocab_size)
model.load_state_dict(torch.load("gpt_model.pth"))
model.eval()

# 生成测试
seed_text = "今天天气晴朗，"
idx = torch.tensor([stoi[ch] for ch in seed_text], dtype=torch.long).unsqueeze(0)
generated = model.generate(idx, max_new_tokens=100)[0]

print("生成的文本：")
print(''.join([itos[i] for i in generated]))
