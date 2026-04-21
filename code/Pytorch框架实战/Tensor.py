"""
    Tensor 张量学习脚本, 笔记脚本不运行
"""

import torch

def dm01():
    # 场景1：标量 张量
    t1 = torch.tensor(10)

    # 场景2：二维列表 张量
    data = [[1,2,3], [4,5,6]]
    t2 = torch.tensor(data)

