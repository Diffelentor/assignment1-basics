import torch

def generate(
    model, 
    tokenizer, 
    prompt: str, 
    max_new_tokens: int = 50, 
    temperature: float = 1.0, 
    top_p: float = 0.9,
    device: str = "cuda"
):
    """
    从语言模型中生成文本
    
    Args:
        model: 训练好的语言模型 (nn.Module)，输入 (batch, seq_len) -> 输出 (batch, seq_len, vocab_size)
        tokenizer: 负责 string <-> token_ids 的工具
        prompt (str): 输入提示文本
        max_new_tokens (int): 最大生成 token 数
        temperature (float): softmax 温度，越小越确定性
        top_p (float): nucleus sampling 的阈值
        device (str): "cuda" 或 "cpu"
    """
    model.eval()
    token_ids = tokenizer.encode_tensor(prompt, device)  # (seq_len, )
    # token_ids = torch.tensor(token_ids).to(device) # (1, seq_len)
    if token_ids.dim() == 1:
        token_ids = token_ids.unsqueeze(0)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            # 1. 前向传播，取最后一个位置的 logits，logits[i,j,k]代表第i个batch，第j+1个位置token_id==k的概率
            logits = model(token_ids)  # (1, seq_len, vocab_size) seq_len是变长的，也就是说LM的输入是变长的，不仅会有prompt还会有根据prompt获取的输出
            logits = logits[:, -1, :]  # (1, vocab_size) 取最后一个logits来计算下一个token对应token_id的概率

            # 2. 温度缩放
            logits = logits / temperature

            # 3. softmax -> 概率分布
            logits = torch.softmax(logits, dim=-1)  # (1, vocab_size)

            # 4. nucleus (top-p) 采样
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_logits = torch.cumsum(sorted_logits, dim=-1) # 累计和

            # 找到累计概率 >= top_p 的最小集合
            cutoff = cumulative_logits >= top_p
            cutoff[..., 1:] = cutoff[..., :-1].clone()  # 右移一个，确保除了第一个 True 之外的后续位置都是 True
            cutoff[..., 0] = 0  # 确保至少有一个 token 被保留

            # 屏蔽掉超出 nucleus 集合的 token
            sorted_logits = sorted_logits.masked_fill(cutoff, 0.0)

            # 归一化
            sorted_logits = sorted_logits / sorted_logits.sum(dim=-1, keepdim=True)

            # 按照概率采样
            next_token_id = torch.multinomial(sorted_logits, num_samples=1)  # 根据概率分布采样，总概率=1
            next_token_id = sorted_indices.gather(-1, next_token_id)  # 从排序索引映射回原始 vocab

            # 5. 拼接到输入序列
            token_ids = torch.cat([token_ids, next_token_id], dim=-1)

            # 如果采样到 <|endoftext|> 就停止
            if next_token_id.item() == tokenizer.eos_token_id:
                break

    return tokenizer.decode(token_ids, skip_special_tokens=True)
