import torch
from core.base_processor import BaseProcessor
import random


class Flatten(BaseProcessor):
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
            # 随机选择展平维度范围
            start_dim = random.randint(0, k - 2)
            end_dim = random.randint(start_dim + 1, k - 1)

        elif self.mode == "increase":
            # 递增维度大小
            arr = [size + self.times * self.step_increment for size in self.dim_sizes]
            # 使用固定维度范围
            start_dim = self.start_dim
            end_dim = self.end_dim
            self.times += 1  # 更新计数器

        else:
            raise NotImplementedError(f"Unsupported mode: {self.mode}")

        self.config = {
            "tensor_shape": arr,
            "start_dim": start_dim,
            "end_dim": end_dim,
            "device": self.device,
        }
        return self.config

    def setup(self):
        tensor_shape = self.config["tensor_shape"]
        self.input_tensor = torch.randn(
            tensor_shape,
            dtype=torch.float,
            device=self.device)

    def execute(self):
        return torch.flatten(
            self.input_tensor,
            start_dim=self.config["start_dim"],
            end_dim=self.config["end_dim"]
        )