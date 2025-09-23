import torch
from torch import nn
from einops import rearrange, einsum
from typing import Dict, List, Tuple
from jaxtyping import Bool, Float, Int
from torch import Tensor

"""在讲义中，线性层在最后一步将d_model维度映射到vocab_size维度"""
class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, weights: torch.Tensor | None = None, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.dtype = dtype

        self.W = nn.Parameter(torch.empty(self.out_features, self.in_features, device=self.device, dtype=self.dtype))
        # self.b = nn.Parameter(torch.empty(out_features, device=device, dtype=dtype))

        if weights is not None:
            # self.W.data.copy_(weights)
            self.load_state_dict({"W":weights})
        # 对权重进行Xavier初始化
        else:
            std = ( 2 / (self.in_features + self.out_features) ) ** 0.5
            torch.nn.init.trunc_normal_(self.W, std=std, a = -3 * std, b = 3 * std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum( x, self.W, " ... d_in, d_out d_in -> ... d_out" )
    
"""在讲义中，embedding层在第一步将token_ids映射到d_model维度"""
class Embedding(nn.Module): # embedding_matrix作为token_id的扩展向量，当作input进入训练，即用向量化的token_id替换原始token_id，方便比较token直接的相似度。最终还是要转化成token_id的形式输出
    def __init__(self, num_embeddings: int, embedding_dim: int, weights: torch.Tensor | None = None, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.num_embeddings = num_embeddings # 词表vocab_size大小
        self.embedding_dim = embedding_dim # 词向量维度d_model
        self.device = device
        self.dtype = dtype

        self.embedding_matrix = nn.Parameter(torch.empty(self.num_embeddings, self.embedding_dim, device=self.device, dtype=self.dtype))
        if weights is not None:
            # self.embedding_matrix.data.copy_(weights)
            self.load_state_dict({"embedding_matrix":weights})
        # 对权重进行初始化
        else:
            std = 1
            torch.nn.init.trunc_normal_(self.embedding_matrix, std=std, a = -3 * std, b = 3 * std)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding_matrix[token_ids] # 从词表到词向量的映射的神经网络里读取token_ids输出词向量
    
class RMSNorm(nn.Module):
    """
    RMSNorm 是归一化技术，它通过将输入除以输入的平方根的平均值来稳定训练。
    公式是：
    x_norm = x / (x.pow(2).mean(dim=-1, keepdim=True) + eps).sqrt()
    x_norm = x_norm * weight
    
    Args:
        d_model (int): 经过embedding层之后，每个token的维度
        eps (float): 一个很小的常数，用于避免除以零
        device (torch.device): 设备
        dtype (torch.dtype): 数据类型
    input:
        x: (batch_size, seq_len, d_model) 输入的稠密向量
    output:
        x_norm: (batch_size, seq_len, d_model) 归一化后的稠密向量
    """
    def __init__(self, d_model: int, eps: float = 1e-5, weights: torch.Tensor | None = None, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype)) #weight对应缩放参数 gamma
        if weights is not None:
            # self.weight.data.copy_(weights)
            self.load_state_dict({"weight":weights})

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 题目要求对于不同的精度要先转换为float32再进行归一化，最后再转换回原来的精度
        input_dtype = x.dtype
        x = x.to(torch.float32)

        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps) # x = x / torch.sqrt(variance + self.eps)

        return (self.weight * x).to(input_dtype)
    
class SwiGLU(nn.Module):
    """
    SwiGLU 是激活函数，它通过将输入乘以sigmoid函数，然后乘以一个线性变换来得到输出。
    公式是：
    out = w2(w1(x) * sigmoid(w1(x)) * w3(x))
    其中x是输入，w1(x)是线性变换，sigmoid(w1(x))是sigmoid函数，w2(x)是线性变换，w3(x)是线性变换。
    Args:
        d_model (int): 输入的维度
        d_ff (int): 输出的维度
    input:
        x: (batch_size, seq_len, d_model) 
    output:
        out: (batch_size, seq_len, d_model) 
    """
    def __init__(self, d_model, d_ff, w1_weight, w2_weight, w3_weight):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.w1 = nn.Linear(d_model, d_ff, bias=False) #注意讲义上是Wx这种列向量的形式出现，简单写法self.w1(x)就是按照行向量了，因此要形状反过来。
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)
        if w1_weight is not None:
            # self.w1.weight.data.copy_(w1_weight)
            self.w1.load_state_dict({"weight":w1_weight})
        if w2_weight is not None:
            # self.w2.weight.data.copy_(w2_weight)
            self.w2.load_state_dict({"weight":w2_weight})
        if w3_weight is not None:
            # self.w3.weight.data.copy_(w3_weight)
            self.w3.load_state_dict({"weight":w3_weight})

    def silu(self, x):
        return x * torch.sigmoid(x)
    
    def forward(self, x):
        return self.w2(self.silu(self.w1(x)) * self.w3(x))
    
class RoPE(nn.Module):
    """
    RoPE 是旋转位置编码，它通过将输入的稠密向量旋转来稳定训练。
    公式是：
    out = x * cos(theta * position) - x * sin(theta * position)
    Args:
        theta (float): 底数超参数
        d_k (int): 输入的维度，也就是d_model
        max_seq_len (int): 最大序列长度
        device (torch.device): 设备
    input:
        x: (batch_size, seq_len, d_model) 输入的稠密向量
        token_positions: (batch_size, seq_len) 每个token的位置信息
    output:
        out: (batch_size, seq_len, d_model) 输出的稠密向量
    """
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        if d_k % 2 != 0:
            raise ValueError("d_k must be even")
        self.theta = theta #这个是RoPE的底数超参数，不是直接的角度
        self.d_k = d_k #d_k就是d_model,即嵌入之后的稠密向量，它必须为偶数
        self.max_seq_len = max_seq_len
        self.device = device
        #计算频率
        freqs = 1.0 / (self.theta ** (torch.arange(0, self.d_k, 2).float() / self.d_k))
        #记录每个token的位置信息，公式中的i
        positions = torch.arange(self.max_seq_len)
        #计算正弦和余弦，theta_{i,k}
        sinusoids = torch.outer(positions, freqs) #outer是外积，即每个位置都与每个频率相乘 shape: [max_seq_len, d_k//2]
        self.register_buffer("cos_cache", sinusoids.cos(), persistent=False) #利用register_buffer表示这是固定的，不需要学习
        self.register_buffer("sin_cache", sinusoids.sin(), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None=None) -> torch.Tensor:
        if token_positions is None:
            token_positions = torch.arange(x.shape[-2],device=x.device)
        # 这里的x是输入的稠密向量，token_positions是token的位置信息
        cos = self.cos_cache[token_positions]
        sin = self.sin_cache[token_positions]

        # cos = cos.unsqueeze(0) # shape: [1, max_seq_len, d_k//2] 对应 [batch, max_seq_len, d_k//2]
        # sin = sin.unsqueeze(0) # shape: [1, max_seq_len, d_k//2] 对应 [batch, max_seq_len, d_k//2]

        #  这里还是分奇偶数写容易理解
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]

        rotated_even = x_even * cos - x_odd * sin # 偶数位置乘以cos，奇数位置乘以sin
        rotated_odd = x_even * sin + x_odd * cos # 偶数位置乘以sin，奇数位置乘以cos

        # # out = torch.cat([output1, output2], dim=-1) # shape: [batch,  max_seq_len, d_k]
        # out = torch.stack([output1, output2], dim=-1)  # [batch, seq_len, d_k//2, 2] #用stack能巧妙的把奇数和偶数交叉在一起，cat就不行
        # out = out.flatten(-2)  # [batch, seq_len, d_k]
        out = torch.zeros_like(x)
        out[..., 0::2] = rotated_even
        out[..., 1::2] = rotated_odd
        return out
    

class ScaledDotProductAttention(nn.Module):
    """
    ScaledDotProductAttention 是缩放点积注意力，它通过将输入的稠密向量与输入的稠密向量进行点积来得到输出。
    公式是：
    out = softmax(QK^T / sqrt(d_k))V
    Args:
        Q: (batch_size, seq_len, d_k) 查询向量
        K: (batch_size, seq_len, d_k) 键向量
        V: (batch_size, seq_len, d_v) 值向量
        mask: (batch_size, seq_len, seq_len) 掩码
    output:
        out: (batch_size, seq_len, d_k) 输出的稠密向量
    """
    def __init__(self):
        super().__init__()

    def forward(
        self, 
        Q: Float[Tensor, " ... queries d_k"],
        K: Float[Tensor, " ... keys d_k"],
        V: Float[Tensor, " ... values d_v"],
        mask: Bool[Tensor, " ... queries keys"] | None = None,
    ) -> Float[Tensor, " ... queries d_v"]:
        d_k = Q.shape[-1]
        scores = einsum( Q, K, "... queries d_k, ... keys d_k -> ... queries keys") / (d_k**0.5)
        # scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k))
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf')) #如果mask为0，则将对应位置的score设置为-inf
        attn_weights = torch.softmax(scores, dim=-1) #对key这一维度进行softmax归一化
        # return torch.matmul(attn_weights, V)
        return einsum( attn_weights, V, "... queries keys, ... keys d_v -> ... queries d_v") # 将attn_weights与value相乘得到最终的输出
    
"""
原始的输入形状是(batch_size, seq_len, d_model)
0.对于输入，要先用W_q, W_k, W_v 线性变换得到q,k,v
1.如果有num_heads个头，就把最后一个维度切分成num_heads份，q,k,v每一部分都切分成 d_model//num_heads 的维度，
2.对于每个头，对于q,k,v都去做attention操作。
3.最后把所有的头按照最后一个维度concat起来，然后做一次线性变换。
"""

class CausalMultiHeadAttention(nn.Module):
    """
    CausalMultiHeadAttention 是因果多头注意力，它通过将输入的稠密向量与输入的稠密向量进行点积来得到输出。
    每个头的公式都是：
    out = softmax(QK^T / sqrt(d_k))V
    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        num_heads (int): Number of heads to use in multi-headed attention.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        q_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the Q projection
        k_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the K projection
        v_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the V projection
        o_proj_weight (Float[Tensor, "d_model d_v"]): Weights for the output projection
        in_features (Float[Tensor, "... sequence_length d_in"]): Tensor to run your implementation on.
    output:
        Float[Tensor, " ... sequence_length d_out"]: Tensor with the output of running your optimized, batched multi-headed attention
        implementation with the given QKV projection weights and input features.
    """
    def __init__(self, d_model, num_heads):
        super(CausalMultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        # self.q_proj = nn.Linear(d_model, d_model, bias=False)
        # self.k_proj = nn.Linear(d_model, d_model, bias=False)
        # self.v_proj = nn.Linear(d_model, d_model, bias=False)
        # self.o_proj = nn.Linear(d_model, d_model, bias=False)

    def attention(
        self, 
        Q: Float[Tensor, " ... queries d_k"],
        K: Float[Tensor, " ... keys d_k"],
        V: Float[Tensor, " ... values d_v"],
        mask: Bool[Tensor, " ... queries keys"] | None = None,
    ) -> Float[Tensor, " ... queries d_v"]:
        d_k = Q.shape[-1]
        scores = einsum( Q, K, "... queries d_k, ... keys d_k -> ... queries keys") / torch.sqrt(torch.tensor(d_k))
        # scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k))
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf')) #如果mask为0，则将对应位置的score设置为-inf
        attn_weights = torch.softmax(scores, dim=-1) #对key这一维度进行softmax归一化
        # return torch.matmul(attn_weights, V)
        return einsum( attn_weights, V, "... queries keys, ... keys d_v -> ... queries d_v") # 将attn_weights与value相乘得到最终的输出
    def forward(self, x, q_proj_weight, k_proj_weight, v_proj_weight, o_proj_weight)->torch.Tensor:
        batch_size, seq_len, _ = x.shape

        # q = x @ q_proj_weight.T # (batch_size, seq_len, d_model) @ (d_model, d_k) -> (batch_size, seq_len, d_k)
        # k = x @ k_proj_weight.T # (batch_size, seq_len, d_model) @ (d_model, d_k) -> (batch_size, seq_len, d_k)
        # v = x @ v_proj_weight.T # (batch_size, seq_len, d_model) @ (d_model, d_v) -> (batch_size, seq_len, d_v)
        # q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        q = einsum( x, q_proj_weight, " batch seq d_in, d_k d_in -> batch seq d_k")
        k = einsum( x, k_proj_weight, " batch seq d_in, d_k d_in -> batch seq d_k")
        v = einsum( x, v_proj_weight, " batch seq d_in, d_v d_in -> batch seq d_v")

        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2) #view会优先切分最后一个维度，这和内存有关。
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # 现在的形状是(batch_size, num_heads, seq_len, head_dim)，即每个batch中，有num_heads个头分别处理seq_len个token，每个token的维度是head_dim
        # 创建mask，用于防止当前位置的token看到未来的token。即第i行可以看到1到i列，i列之后的都是0
        mask = torch.tril(torch.ones(seq_len, seq_len,dtype=torch.bool,device=x.device)) # 取下三角为1，默认diagonal=0，严格下三角线（包含对角线）
        # mask = torch.tril(torch.ones(seq_len, seq_len)).bool()
        # mask = mask.unsqueeze(0).unsqueeze(0) # (1, 1, seq_len, seq_len)


        out = self.attention(q, k, v, mask)
        out = out.transpose(1, 2)
        out = out.contiguous().view(batch_size, seq_len, self.d_model)
        # out = out @ o_proj_weight.T
        out = einsum( out, o_proj_weight, " batch seq d_v, d_model d_v -> batch seq d_model")
        return out
    


class CausalMultiHeadAttentionWithRoPE(nn.Module):
    def __init__(self, d_model:int, num_heads:int, max_seq_len:int, theta:float ,q_proj_weight:torch.Tensor, k_proj_weight:torch.Tensor, v_proj_weight:torch.Tensor, o_proj_weight:torch.Tensor,device=None):
        super(CausalMultiHeadAttentionWithRoPE, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.max_seq_len = max_seq_len
        self.theta = theta
        self.rope = RoPE(theta, self.head_dim, max_seq_len, device) #这里要注意，我们使用多头注意力机制的时候，每个head的维度是d_model // num_heads，我们应当对每个head进行RoPE
        self.q_proj_weight = q_proj_weight
        self.k_proj_weight = k_proj_weight
        self.v_proj_weight = v_proj_weight
        self.o_proj_weight = o_proj_weight
        

    def attention(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor | None = None):
        d_k = Q.shape[-1]
        scores = einsum( Q, K, "... queries d_k, ... keys d_k -> ... queries keys") / torch.sqrt(torch.tensor(d_k))
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf')) #如果mask为0，则将对应位置的score设置为-inf
        attn_weights = torch.softmax(scores, dim=-1) #对key这一维度进行softmax归一化
        return einsum( attn_weights, V, "... queries keys, ... keys d_v -> ... queries d_v") # 将attn_weights与value相乘得到最终的输出
    
    def forward(self, x,  token_positions=None)->torch.Tensor:
        # if token_positions is None:
        #     token_positions = torch.arange(x.shape[1],device=x.device)
        batch_size, seq_len, _ = x.shape

        # q = x @ q_proj_weight.T # (batch_size, seq_len, d_model) @ (d_model, d_k) -> (batch_size, seq_len, d_k)
        # k = x @ k_proj_weight.T # (batch_size, seq_len, d_model) @ (d_model, d_k) -> (batch_size, seq_len, d_k)
        # v = x @ v_proj_weight.T # (batch_size, seq_len, d_model) @ (d_model, d_v) -> (batch_size, seq_len, d_v)
        # q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        q = einsum( x, self.q_proj_weight, " batch seq d_in, d_k d_in -> batch seq d_k")
        k = einsum( x, self.k_proj_weight, " batch seq d_in, d_k d_in -> batch seq d_k")
        v = einsum( x, self.v_proj_weight, " batch seq d_in, d_v d_in -> batch seq d_v")

        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2) #view会优先切分最后一个维度，这和内存有关。
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        q = self.rope(q, token_positions)     
        k = self.rope(k, token_positions)

        mask = torch.tril(torch.ones(seq_len, seq_len,dtype=torch.bool,device=x.device)) # 取下三角为1，默认diagonal=0，严格下三角线（包含对角线）

        out = self.attention(q, k, v, mask)
        out = out.transpose(1, 2)
        out = out.contiguous().view(batch_size, seq_len, self.d_model)
        # out = out @ o_proj_weight.T
        out = einsum(out, self.o_proj_weight, " batch seq d_v, d_model d_v -> batch seq d_model")
        return out

class TransformerBlock(nn.Module):
    """
    TransformerBlock 是Transformer块，它把包含多头注意力机制的一些组件包装在一起，形成一个完整的Transformer块。
    Args:
        d_model (int): 输入的维度，也就是d_model
        num_heads (int): 头的数量
        d_ff (int): 前馈神经网络的维度
        max_seq_len (int): 最大序列长度
        theta (float): 底数超参数
        attn_q_proj_weight (torch.Tensor): 查询的权重
    """
    def __init__(self, d_model:int, num_heads:int, d_ff:int, max_seq_len:int, theta:float, weights: dict[str, Tensor], device=None):
        super(TransformerBlock, self).__init__()
        "权重"
        # self.attn_q_proj_weight = weights["attn.q_proj.weight"]
        # self.attn_k_proj_weight = weights["attn.k_proj.weight"]
        # self.attn_v_proj_weight = weights["attn.v_proj.weight"]
        # self.attn_o_proj_weight = weights["attn.output_proj.weight"]

        # self.ln1_weight = weights["ln1.weight"]
        # self.ln2_weight = weights["ln2.weight"]

        # self.ffn_w1_weight = weights["ffn.w1.weight"]
        # self.ffn_w2_weight = weights["ffn.w2.weight"]
        # self.ffn_w3_weight = weights["ffn.w3.weight"]
        self.weights = weights
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len
        self.theta = theta
        self.device = device
        # self.linear_module = LinearModule(d_model,d_model,device)
        # self.embedding_module = EmbeddingModule(d_model,d_model,device)
        self.rms_norm1 = RMSNorm(d_model,eps=1e-5, weights=self.weights["ln1.weight"], device=device)
        # self.rms_norm1.load_state_dict({"weight":self.ln1_weight})
        self.rms_norm2 = RMSNorm(d_model,eps=1e-5, weights=self.weights["ln2.weight"], device=device)
        # self.rms_norm2.load_state_dict({"weight":self.ln2_weight})
        self.swiglu = SwiGLU(d_model,d_ff, w1_weight=self.weights["ffn.w1.weight"], w2_weight=self.weights["ffn.w2.weight"], w3_weight=self.weights["ffn.w3.weight"])
        # self.swiglu.load_state_dict({"w1.weight":self.ffn_w1_weight,"w2.weight":self.ffn_w2_weight,"w3.weight":self.ffn_w3_weight})
        self.causal_multi_head_attention = CausalMultiHeadAttentionWithRoPE(d_model,num_heads,max_seq_len,theta,self.weights["attn.q_proj.weight"],self.weights["attn.k_proj.weight"],self.weights["attn.v_proj.weight"],self.weights["attn.output_proj.weight"],device)

    def forward(self,in_features:torch.Tensor):
        token_positions = torch.arange(in_features.shape[1],device=in_features.device)
        x1 = self.rms_norm1(in_features)
        x1 = self.causal_multi_head_attention(x1,token_positions)
        x1 = x1 + in_features
        x2 = self.rms_norm2(x1)
        x2 = self.swiglu(x2)
        out = x2 + x1
        return out

class TransformerLM(nn.Module):
    """
    TransformerLM 是整个训练过程的封装，它把包含Embedding、TransformerBlock、RMSNorm、LinearModule等组件包装在一起，形成一个完整的Transformer语言模型。
    Args:
        vocab_size (int): 词表大小
        context_length (int): 上下文长度
        d_model (int): 输入的维度，也就是d_model
        num_layers (int): 层数
        num_heads (int): 头的数量
        d_ff (int): 前馈神经网络的维度
        rope_theta (float): 底数超参数
        weights (dict[str, torch.Tensor]): 权重
    input:
        in_indices (torch.Tensor): 输入的索引
    output:
        out_linear (torch.Tensor): 输出的线性层
    """
    def __init__(self, vocab_size:int, context_length:int, d_model:int, num_layers:int, num_heads:int, d_ff:int, rope_theta:float, weights:dict[str, torch.Tensor]):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.rope_theta = rope_theta
        self.weights = weights
        self.embedding_layer = Embedding(self.vocab_size,self.d_model,device=None, weights=self.weights["token_embeddings.weight"])
        self.transformer_blocks = nn.ModuleList()
        for layer in range(self.num_layers):
            prefix = f"layers.{layer}."
            layer_weights = {k[len(prefix):]: v for k, v in self.weights.items() if k.startswith(prefix)}
            transformer_block = TransformerBlock(self.d_model,self.num_heads,self.d_ff,self.context_length,self.rope_theta,layer_weights,device=None)
            self.transformer_blocks.append(transformer_block)
        self.norm_layer = RMSNorm(self.d_model,eps=1e-5,weights=self.weights["ln_final.weight"],device=None)
        self.linear_layer = Linear(self.d_model,self.vocab_size,weights=self.weights["lm_head.weight"],device=None)
        

    def forward(self, in_indices):
        embedding = self.embedding_layer(in_indices)

        for transformer_block in self.transformer_blocks:
            embedding = transformer_block(embedding)
            
        transformer_block_out = embedding   

        out_norm = self.norm_layer(transformer_block_out)

        out_linear = self.linear_layer(out_norm)
        
        return out_linear








    
