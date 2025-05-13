from torch import nn

from core.base_processor import BaseProcessor
import random
import torch


class LayerNorm(BaseProcessor):
    def __init__(self, args, logger):
        super().__init__(args, logger)
        self.input_tensor = None
        self.norm_layer = None

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

        # 随机选择从哪个维度开始normalize
        dim = random.randint(0, k - 1)
        normalized_shape = arr[dim:]
        self.config = {
            "tensor_shape": arr,
            "normalized_shape": normalized_shape,
        }
        return self.config

    def setup(self):
        tensor_shape = self.config["tensor_shape"]
        normalized_shape = self.config["normalized_shape"]
        self.norm_layer = nn.LayerNorm(normalized_shape).to(self.device)
        self.input_tensor = torch.randn(
            tensor_shape,
            dtype=torch.float,
            device=self.device,
            pin_memory=False
        )

    def execute(self):
        return self.norm_layer(self.input_tensor)
