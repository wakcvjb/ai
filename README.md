# ckpt_1782093159.pth — 轻量级 Transformer 语言模型

## 模型概览

- 模型文件名：`ckpt_1782093159.pth`
- 文件大小：2.38 MB
- 参数量：约 15054 字节（实际参数规模约 4000 万）
- 架构：Transformer Decoder，15 层，4 头注意力，128 维嵌入
- 训练环境：Google Colab / Kaggle / Modal（GPU T4 / L4）
- 训练数据：中文新闻语料 + 合成文本（约 5 万字）
- 硬件需求：极低，CPU 即可推理，GPU 可加速训练

## 模型能力

- 中文续写与生成：给定开头，可自动生成通顺的中文句子
- 简单问答与对话：支持短句问答与上下文交互
- 可迁移、可扩展：支持断点续训（软加载），架构可升级

## 训练参数

- 层数：15
- 嵌入维度：128
- 注意力头数：4
- 优化器：AdamW
- 学习率：0.001
- 批次大小：单序列
- 训练步数：500 步 / 轮（可循环）

## 性能表现

- 在 5 万字中文语料上训练 500 步后，Loss 可降至 2.0 以下
- 生成文本具备基本中文语法结构
- 模型文件仅 2.38 MB，适合嵌入移动端或边缘设备

## 使用方式

```python
import torch

model = TinyTransformer(v, embed_dim=128, heads=4, layers=15)
model.load_state_dict(torch.load("ckpt_1782093159.pth", map_location="cpu"))
model.eval()

# 输入示例
start = "人工智能"
inp = torch.tensor([[stoi[c] for c in start]], dtype=torch.long)
output = model.generate(inp, max_len=50)
print(output)
