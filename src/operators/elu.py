import torch
from core.base_processor import BaseProcessor
import random
class ELU(BaseProcessor):
    def __init__(self, args, logger):
        super().__init__(args, logger)
        self.alpha = 0.1  # 控制负值区域的缩放因子
        self.input_tensor = None
        self.output_tensor = None
        self.current_index = 0
    def generate_config(self):
        MAX_DIM_NUM = int(self.MAX_DIM_NUM)
        MIN_DIM_NUM = int(self.MIN_DIM_NUM)
        SINGLE_DIM_LENGTH_MAX = int(self.SINGLE_DIM_LENGTH_MAX)
        SINGLE_DIM_LENGTH_MIN = int(self.SINGLE_DIM_LENGTH_MIN)
        k = random.randint(MIN_DIM_NUM, MAX_DIM_NUM)
        arr = [self.BASIC_NUMBER for _ in range(k)]

        if self.mode == "random":
            arr = [random.randint(SINGLE_DIM_LENGTH_MIN, SINGLE_DIM_LENGTH_MAX) for _ in range(k)]
            self.config = {"tensor_shape": arr}
        elif self.mode == "increase":
            arr[self.current_index] += self.BASIC_NUMBER
            if arr[self.current_index] > int(SINGLE_DIM_LENGTH_MAX):
                arr[self.current_index] = int(SINGLE_DIM_LENGTH_MAX)
            self.current_index = (self.current_index + 1) % k
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
