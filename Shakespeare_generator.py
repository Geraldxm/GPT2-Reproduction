import torch
import torch.nn as nn
from torch.nn import functional as F
# from Shakespeare_gpt import Head, MultiHeadAttention, FeedFoward, Block, GPTLanguageModel
from Shakespeare_gpt import *

# 加载模型
model = GPTLanguageModel()
print('Model has {:,} parameters'.format(sum(p.numel() for p in model.parameters())))

model.load_state_dict(torch.load('weights/best_240728.pt'))

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)  # 确保模型在相同的设备上
with torch.no_grad():
    model.eval()  # 设置模型为评估模式

# 使用 "ROMEO:" 为开头 generate
context = torch.tensor(encode("ROMEO:"), dtype=torch.long, device=device).unsqueeze(0)
print('context shape:', context.shape)
generated_tokens = model.generate(context, max_new_tokens=1000)
print('generated tokens shape:', generated_tokens.shape)
generated_text = decode(generated_tokens[0].tolist())
print(generated_text)
