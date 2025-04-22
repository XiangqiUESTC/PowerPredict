import torch
from core.base_processor import BaseProcessor
import random

class Flatten(BaseProcessor):
    def __init__(self):
        super().__init__()
        self.input_tensor = None
    def generate_config(self):
        # 最大维数
        MAX_DIM_NUM = 4
        # 最小维数
        MIN_DIM_NUM = 2
        # 每个维度的区间
        SINGLE_DIM_LENGTH_MAX = 256
        SINGLE_DIM_LENGTH_MIN = 1
        k = random.randint(MIN_DIM_NUM, MAX_DIM_NUM)  # 随机维度数量
        arr = [random.randint(SINGLE_DIM_LENGTH_MIN, SINGLE_DIM_LENGTH_MAX) for _ in range(k)]  # 生成维度值
        # 展平的开始维度
        start_dim = random.randint(0, k - 2)
        end_dim = random.randint(start_dim+1, k - 1)
        self.config = {
            "tensor_shape": arr,
            "start_dim": start_dim,
            "end_dim": end_dim,
        }
        return self.config

    def setup(self):
        tensor_shape = self.config["tensor_shape"]
        # 生成tensor
        self.input_tensor = torch.randn(
            tensor_shape,
            dtype=torch.float,
            device="cuda")
    def execute(self):
        return torch.flatten(self.input_tensor, start_dim=self.config["start_dim"], end_dim=self.config["end_dim"])
