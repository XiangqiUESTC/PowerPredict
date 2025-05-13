from core.base_processor import BaseProcessor
import random
import torch


class Add(BaseProcessor):
    def __init__(self, logger):
        super().__init__(logger)
        self.A = None
        self.B = None

    def generate_config(self):
        # 最大维数
        MAX_DIM_NUM = 4
        # 最小维数
        MIN_DIM_NUM = 1
        # 每个维度的区间
        SINGLE_DIM_LENGTH_MAX = 256
        SINGLE_DIM_LENGTH_MIN = 1

        # 随机维度数量
        k = random.randint(MIN_DIM_NUM, MAX_DIM_NUM)
        # 生成维度值
        arr = [random.randint(SINGLE_DIM_LENGTH_MIN, SINGLE_DIM_LENGTH_MAX) for _ in range(k)]
        # 生成配置字典
        self.config = {
            "tensor_shape": arr
        }
        return self.config

    def setup(self):
        tensor_shape = self.config["tensor_shape"]
        self.A = torch.randn(
            tensor_shape,
            dtype=torch.float,
            device=self.device,
            pin_memory=False
        )
        self.B = torch.randn(
            tensor_shape,
            dtype=torch.float,
            device=self.device,
            pin_memory=False
        )

    def execute(self):
        return torch.add(self.A, self.B)
