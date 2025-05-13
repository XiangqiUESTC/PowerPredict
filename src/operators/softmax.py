import torch

from core.base_processor import BaseProcessor
import random


class Softmax(BaseProcessor):
    def __init__(self, args, logger):
        super().__init__(args, logger)
        self.input_tensor = None

    def generate_config(self):
        # 最大维数
        MAX_DIM_NUM = 3
        # 最小维数
        MIN_DIM_NUM = 1
        # 每个维度的区间
        SINGLE_DIM_LENGTH_MAX = 1024
        SINGLE_DIM_LENGTH_MIN = 1

        # 随机维度数量
        k = random.randint(MIN_DIM_NUM, MAX_DIM_NUM)
        # 生成维度值
        arr = [random.randint(SINGLE_DIM_LENGTH_MIN, SINGLE_DIM_LENGTH_MAX) for _ in range(k)]
        # 随机选择操作的维度
        dim = random.randint(0, k - 1)
        # 返回配置字典
        self.config = {"tensor_shape": arr, "dim": dim}
        return self.config

    def setup(self):
        tensor_shape = self.config["tensor_shape"]
        self.input_tensor = torch.randn(
            tensor_shape,
            dtype=torch.float,
            device=self.device,
            pin_memory=False
        )

    def execute(self):
        return torch.softmax(self.input_tensor, self.config["dim"])


