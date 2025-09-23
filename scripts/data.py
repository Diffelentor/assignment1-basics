import numpy as np
import numpy.typing as npt
import torch
from typing import Tuple
import os

def get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    随机获取一个batch的数据，y 正好是 x 向左移动一个位置的结果。模型在看到 x 的第 i 个位置的输入时，需要努力预测出 y 在第 i 个位置的词元，这正是我们想要的“预测下一个词”的效果。
    """
    idxs = np.random.randint(0, len(dataset)-context_length, size=(batch_size,)) # 随机选择一个batch_size大小的索引,最后的括号其实写不写都行，写上了就更明确输出是一个数组
    x = np.stack([dataset[id:id+context_length] for id in idxs])
    y = np.stack([dataset[id+1:id+context_length+1] for id in idxs])
    return torch.from_numpy(x).to(device),torch.from_numpy(y).to(device)

class Dataset:
    def __init__(self, dataset_name: str, context_length: int, batch_size: int, device: str, **kwargs):
        dataset_path = os.path.join("data", dataset_name)
        self.train_data = np.memmap(os.path.join(dataset_path, "train.bin"), dtype=np.uint16, mode='r').astype(np.int64)
        self.val_data = np.memmap(os.path.join(dataset_path, "val.bin"), dtype=np.uint16, mode='r').astype(np.int64)
        self.context_length = context_length
        self.batch_size = batch_size
        self.device = device
    
    def get_batch(self, split: str) -> Tuple[torch.Tensor, torch.Tensor]:
        data = self.train_data if split == 'train' else self.val_data
        return get_batch(data, self.batch_size, self.context_length, self.device)
    
    def get_train_data(self):
        return self.train_data
    
    def get_val_data(self):
        return self.val_data
    
    def __len__(self):
        return len(self.train_data) if hasattr(self, "train_data") else 0

    def __getitem__(self, idx):
        # 默认从 train_data 取，可以加个参数控制
        x = self.train_data[idx: idx+self.context_length]
        y = self.train_data[idx+1: idx+self.context_length+1]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)
