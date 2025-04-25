import torch
from core.base_processor import BaseProcessor
import random


class SiLU(BaseProcessor):
    def __init__(self):
        super().__init__()
        self.input_tensor = None
        self.output_tensor = None

    def generate_config(self):
        MAX_DIM_NUM = 4
        MIN_DIM_NUM = 1
        SINGLE_DIM_LENGTH_MAX = 512
        SINGLE_DIM_LENGTH_MIN = 1

        # 生成随机维度的张量形状
        k = random.randint(MIN_DIM_NUM, MAX_DIM_NUM)
        arr = [random.randint(SINGLE_DIM_LENGTH_MIN, SINGLE_DIM_LENGTH_MAX) for _ in range(k)]
        self.config = {"tensor_shape": arr}

    def setup(self):
        self.input_tensor = torch.tensor(
            self.config["tensor_shape"],
            dtype=torch.float,
            device=self.device
        )

    def execute(self):
        # SiLU是逐元素操作，无需指定维度
        self.output_tensor = torch.nn.SiLU(self.input_tensor)
        return self.output_tensor
