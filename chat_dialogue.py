import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"设备: {device}")

data_file = "sanguo_online.txt"
if not os.path.exists(data_file):
    print(f"找不到 {data_file}")
    exit()

with open(data_file, "r", encoding="utf-8", errors="ignore") as f:
    full_text = f.read()

sample_text = full_text[:20000]
chars = sorted(set(sample_text))
v = len(chars)
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for i, c in enumerate(chars)}

class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, heads, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.heads = heads
        self.head_dim = embed_dim // heads
        self.wq = nn.Linear(embed_dim, embed_dim)
        self.wk = nn.Linear(embed_dim, embed_dim)
        self.wv = nn.Linear(embed_dim, embed_dim)
        self.fc = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        b, s, e = x.shape
        q = self.wq(x).view(b, s, self.heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(b, s, self.heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(b, s, self.heads, self.head_dim).transpose(1, 2)
        score = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        mask = torch.tril(torch.ones(s, s)).view(1, 1, s, s).to(x.device)
        score = score.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(score, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v).transpose(1, 2).reshape(b, s, e)
        return self.fc(out)

class FeedForward(nn.Module):
    def __init__(self, embed_dim, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, embed_dim * 4)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(embed_dim * 4, embed_dim)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        return self.fc2(self.dropout(self.act(self.fc1(x))))

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, heads, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadAttention(embed_dim, heads, dropout)
        self.ff = FeedForward(embed_dim, dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x

class TinyTransformer(nn.Module):
    def __init__(self, v, embed_dim=64, heads=4, layers=4, dropout=0.1):
        super().__init__()
        self.embed = nn.Embedding(v, embed_dim)
        self.pos_enc = PositionalEncoding(embed_dim)
        self.blocks = nn.Sequential(*[TransformerBlock(embed_dim, heads, dropout) for _ in range(layers)])
        self.norm = nn.LayerNorm(embed_dim)
        self.fc = nn.Linear(embed_dim, v)
    def forward(self, x):
        x = self.embed(x)
        x = self.pos_enc(x)
        return self.fc(self.norm(self.blocks(x)))

model_file = "V1_1782372942.pth"
if not os.path.exists(model_file):
    print(f"找不到模型: {model_file}")
    exit()

print("加载模型中...")
model = TinyTransformer(v).to(device)
model.load_state_dict(torch.load(model_file, map_location=device))
model.eval()
print("模型已加载。输入 'exit' 或 'quit' 退出。")

while True:
    user_input = input("\n你: ")
    if user_input.lower() in ["exit", "quit"]:
        break

    inp = torch.tensor([[stoi.get(c, 0) for c in user_input]], dtype=torch.long).to(device)

    with torch.no_grad():
        for _ in range(50):
            logits = model(inp)
            probs = F.softmax(logits[0, -1, :], dim=-1)
            nxt = torch.multinomial(probs, 1).item()
            if nxt < 0 or nxt >= v:
                nxt = 0
            print(itos[nxt], end="")
            inp = torch.cat([inp, torch.tensor([[nxt]], dtype=torch.long).to(device)], dim=1)
    print()
