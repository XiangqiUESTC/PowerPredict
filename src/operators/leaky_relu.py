import torch
from core.base_processor import BaseProcessor
import random


class Leaky_ReLu(BaseProcessor):
    def __init__(self, logger):
        super().__init__(logger)
        self.input_tensor = None
        self.output_tensor = None

    def generate_config(self):
        MAX_DIM_NUM = 4
        MIN_DIM_NUM = 1
        SINGLE_DIM_LENGTH_MAX = 512
        SINGLE_DIM_LENGTH_MIN = 1
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
        # LeakyReLU是逐元素操作，无需指定维度
        # 负斜率设为默认值0.01（可调整）
        self.output_tensor = torch.nn.functional.leaky_relu(self.input_tensor, negative_slope=0.01)
        return self.output_tensor