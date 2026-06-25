# 轻量级 Transformer 语言模型

本项目是一个基于《三国演义》文本训练的微型 Transformer 语言模型。模型文件仅有 1.8 MB，可在手机端或电脑端本地 CPU 环境下运行。

## 包含文件
- `chat_dialogue.py`：加载模型并进行交互式对话的程序。
- `ckpt_1782093159.pth`：训练好的模型权重文件。

## 运行与环境配置 (手机/电脑通用)

将 `chat_dialogue.py` 和 `ckpt_1782093159.pth` 放入同一个文件夹。打开命令行终端并进入该文件夹，依次执行以下命令：

pkg update -y
pkg upgrade -y
pkg install python -y
pkg install python-torch -y
pip install torch
python chat_dialogue.py

手机端使用 Termux 运行以上所有命令。电脑端不需要执行 pkg 相关命令，仅需执行 pip install torch 和 python chat_dialogue.py。

## 使用建议

由于模型仅基于《三国演义》文本训练，建议输入三国人物名、地名或文言句式进行交互，例如：刘备、曹操、赤壁之战、孔明曰。该模型属于小型神经网络的单轮续写，不具备上下文记忆能力，生成的内容可能存在跳跃或不通顺的情况，属于正常现象。输入 exit 或 quit 可退出程序。

## 更新日志

2026-06-25
- 发布初始版本。
- 支持在手机端 Termux 和电脑端 Windows / macOS / Linux 运行。
- 实现交互式对话功能。
- 提供基于《三国演义》前 2 万字训练的模型权重文件。
