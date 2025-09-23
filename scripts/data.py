import numpy as np
import numpy.typing as npt
import torch

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
