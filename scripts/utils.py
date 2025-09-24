import torch
from torch import nn

def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    """
    softmax 是归一化技术，它通过将输入除以输入的指数的平均值来稳定训练。
    公式是：
    out = exp(x - x_max) / sum(exp(x - x_max)) 
    Args:
        x: (batch_size, seq_len, d_model) 输入的稠密向量
        dim: 归一化的维度
    output:
        out: (batch_size, seq_len, d_model) 归一化后的稠密向量
    """
    x_max,_ = x.max(dim=dim, keepdim=True) # 这里x_max是batch_size, seq_len, 1，且这里返回的是 values, indices
    x_exp = torch.exp(x - x_max)
    return x_exp / x_exp.sum(dim=dim, keepdim=True)

def log_softmax(x, dim=-1):
    x_max,_ = x.max(dim=dim, keepdim=True) # 这里x_max是batch_size, seq_len, 1，且这里返回的是 values, indices
    x_exp = torch.exp(x - x_max)
    return x-x_max - torch.log(x_exp.sum(dim=dim, keepdim=True))

class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, inputs, targets):
        y = log_softmax(inputs,-1) # 要在softmax中途就log，保证计算准确性
        y_select = y[torch.arange(inputs.shape[0]),targets] # 选出每行对应target类别的值
        return - y_select.mean()

   
class GradientClip:
    def __init__(self, parameters, max_l2_norm, eps=1e-6):
        self.parameters = parameters
        self.max_l2_norm = max_l2_norm
        self.eps = eps

    def __call__(self):
        grads = [p.grad for p in self.parameters if p.grad is not None] #我们求l2范数是对所有元素求的，所以要先把所有元素给flatten
        grads_flatten = torch.cat([grad.flatten() for grad in grads]) #把所有梯度展平并连接成一个向量
        # grads_flatten = torch.cat([grad.view(-1) for grad in grads])
        grads_l2 = torch.norm(grads_flatten,2)
        if grads_l2 > self.max_l2_norm:
            clip_coef = self.max_l2_norm / (grads_l2 + self.eps)
            for grad in grads:
                grad.mul_(clip_coef)