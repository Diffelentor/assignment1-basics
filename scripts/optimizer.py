import torch
from torch import nn
from collections.abc import Callable, Iterable
from typing import Optional
import math
from torch.optim.lr_scheduler import _LRScheduler

class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)
    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        
        for group in self.param_groups:
            lr = group["lr"] # Get the learning rate.
            for p in group["params"]:
                if p.grad is None:
                    continue
                # 获取该参数的状态字典（可以存储迭代次数 t 等信息）
                state = self.state[p] # Get state associated with p.
                t = state.get("t", 0) # 获取迭代次数，默认 0
                grad = p.grad.data # 获取当前梯度
                p.data-= lr / torch.sqrt(t + 1) * grad # 参数更新：使用逐步衰减学习率 lr / sqrt(t+1)
                state["t"] = t + 1 # 更新迭代次数
        return loss

class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999),
                 eps=1e-8, weight_decay=1e-2):
        # 检查超参数合法性
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super(AdamW, self).__init__(params, defaults)

    def step(self):
        for group in self.param_groups: # 以相同超参数对应的分类
            for p in group['params']: # 同一个超参数组下的不同网络层
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]
                
                if len(state) == 0:
                    state['step'] = 0
                    # exp_avg = 一阶动量
                    state['m'] = torch.zeros_like(p)
                    # exp_avg_sq = 二阶动量
                    state['v'] = torch.zeros_like(p)
                m = state['m']
                v = state['v']
                # m = state.get("m", torch.zeros_like(p)) # 如果state中没有m，就创建一个和p同形状的0张量
                # v = state.get("v", torch.zeros_like(p)) # 如果用get则不是引用，修改后需要重新回去赋值 state['v'] = v
                beta1, beta2 = group['betas']
                state['step'] +=1
                
                # Adam更新
                m.mul_(beta1).add_(grad, alpha=1 - beta1) #mul_ 和 add_ 都是inplace操作，就地修改
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                # 偏差修正
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1 # 标量math.sqrt比**0.5快
                
                denom = v.sqrt().add_(group['eps']) 
                # p.data.addcdiv_(-step_size, m, denom) #+先乘后除
                p.data.addcdiv_(m, denom, value=-step_size)
                # p.data.addcdiv_(m, denom, value=-step_size)

                # 解耦权重衰减,直接减，不要修改梯度
                p.data.add_(p.data, alpha=-group['weight_decay'] * group['lr'])
                # p.data.mul_(1 - group['lr'] * group['weight_decay'])
                
                
    
class CosineSchedule:
    def __init__(self, max_learning_rate, min_learning_rate, warmup_iters, cosine_cycle_iters):
        self.max_learning_rate = max_learning_rate
        self.min_learning_rate = min_learning_rate
        self.warmup_iters = warmup_iters
        self.cosine_cycle_iters = cosine_cycle_iters

    def __call__(self, it):
        """
        it: 当前迭代次数
        """
        if it < self.warmup_iters:
            return it * self.max_learning_rate / self.warmup_iters
        elif it > self.cosine_cycle_iters:
            return self.min_learning_rate
        else:
            return self.min_learning_rate + 0.5 * (1 + math.cos(math.pi * (it - self.warmup_iters) / (self.cosine_cycle_iters - self.warmup_iters))) * (self.max_learning_rate - self.min_learning_rate)
        

class CosineScheduleLR(_LRScheduler):
    """
    Cosine learning rate scheduler with linear warmup and optional min_lr floor.
    Compatible with PyTorch's lr_scheduler interface.
    """
    def __init__(self, optimizer, max_lr, min_lr, warmup_iters, cosine_cycle_iters, last_iter=-1):
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.warmup_iters = warmup_iters
        self.cosine_cycle_iters = cosine_cycle_iters
        super().__init__(optimizer, last_iter)

    def get_lr(self):
        # last_iter 在 PyTorch 中就是迭代步数
        if self.last_epoch < self.warmup_iters:
            lr = self.last_epoch * self.max_lr / self.warmup_iters
        elif self.last_epoch > self.cosine_cycle_iters:
            lr = self.min_lr
        else:
            lr = self.min_lr + 0.5 * (1 + math.cos(
                math.pi * (self.last_epoch - self.warmup_iters) / (self.cosine_cycle_iters - self.warmup_iters)
            )) * (self.max_lr - self.min_lr)
        return [lr for _ in self.optimizer.param_groups]  # 返回每个 param_group 的 lr
