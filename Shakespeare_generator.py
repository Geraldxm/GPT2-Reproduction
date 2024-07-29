import torch
import torch.nn as nn
from torch.nn import functional as F
# from Shakespeare_gpt import Head, MultiHeadAttention, FeedFoward, Block, GPTLanguageModel
from Shakespeare_gpt import *

# 加载模型
model = GPTLanguageModel()
print('Model has {:,} parameters'.format(sum(p.numel() for p in model.parameters())))

# 加载最佳模型权重
model.load_state_dict(torch.load('best.pt'))
# 或者加载最后一次的模型权重
# model.load_state_dict(torch.load('last.pt'))

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)  # 确保模型在相同的设备上
model.eval()  # 设置为评估模式

# 定义编码和解码函数
# chars = sorted(list(set(text)))
# vocab_size = len(chars)
# stoi = {ch: i for i, ch in enumerate(chars)}
# itos = {i: ch for i, ch in enumerate(chars)}
# encode = lambda s: [stoi[c] for c in s]  # encoder: take a string, output a list of integers
# decode = lambda l: ''.join([itos[i] for i in l])  # decoder: take a list of integers, output a string

# 使用 "ROMEO:" 为开头进行 generate
context = torch.tensor(encode("ROMEO:"), dtype=torch.long, device=device).unsqueeze(0)
generated_tokens = model.generate(context, max_new_tokens=500)
generated_text = decode(generated_tokens[0].tolist())
print(generated_text)