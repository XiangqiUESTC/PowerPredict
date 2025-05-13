import torch
from core.base_processor import BaseProcessor
import random


class ELU(BaseProcessor):
    def __init__(self, args, logger):
        super().__init__(args, logger)
        self.alpha = alpha  # 控制负值区域的缩放因子
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
        # ELU是逐元素操作，无需指定维度
        self.output_tensor = torch.nn.functional.elu(self.input_tensor, alpha=self.alpha)
        return self.output_tensor
