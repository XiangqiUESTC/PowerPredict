import torch
from core.base_processor import BaseProcessor
import random


class Softmax(BaseProcessor):
    def __init__(self, args, logger):
        super().__init__(args, logger)
        self.input_tensor = None

        # 递增模式初始化
        if self.mode == "increase":
            self.times = 0  # 运行次数计数器
            # 初始化维度大小（随机值在[min_dim_size, max_dim_size]区间）
            self.dim_sizes = [
                random.randint(self.min_dim_size, self.max_dim_size)
                for _ in range(self.dim_num)
            ]

    def generate_config(self):
        if self.mode == "random":
            # 使用yaml配置参数
            k = random.randint(self.min_dim_num, self.max_dim_num)
            arr = [
                random.randint(self.min_dim_size, self.max_dim_size)
                for _ in range(k)
            ]
            dim = random.randint(0, k - 1)

        elif self.mode == "increase":
            # 递增维度大小
            arr = [size + self.times * self.step_increment for size in self.dim_sizes]
            dim = self.start_dim  # 固定计算维度
            self.times += 1  # 更新计数器

        else:
            raise NotImplementedError(f"Unsupported mode: {self.mode}")

        self.config = {"tensor_shape": arr, "dim": dim, "device": self.device}
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