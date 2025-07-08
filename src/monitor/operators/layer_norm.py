from torch import nn
from monitor.core.base_processor import BaseProcessor
import random
import torch


class LayerNorm(BaseProcessor):
    def __init__(self, args, logger):
        super().__init__(args, logger)
        self.input_tensor = None
        self.norm_layer = None

        # 递增模式初始化
        if self.mode == "increase":
            self.times = 0
            # 初始化维度大小
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
            # 随机选择归一化起始维度
            dim = random.randint(0, k - 1)
            normalized_shape = arr[dim:]

        elif self.mode == "increase":
            # 递增维度大小
            arr = [size + self.times * self.step_increment for size in self.dim_sizes]
            # 使用固定归一化起始维度
            dim = self.start_dim
            normalized_shape = arr[dim:]
            self.times += 1

        else:
            raise NotImplementedError(f"Unsupported mode: {self.mode}")

        self.config = {
            "tensor_shape": arr,
            "normalized_shape": normalized_shape,
            "device": self.device,
        }
        return self.config

    def setup(self):
        tensor_shape = self.config["tensor_shape"]
        normalized_shape = self.config["normalized_shape"]

        # 创建并移动LayerNorm层到指定设备
        self.norm_layer = nn.LayerNorm(normalized_shape).to(self.device)

        # 创建输入张量并移动到指定设备
        self.input_tensor = torch.randn(
            tensor_shape,
            dtype=torch.float,
            device=self.device,
            pin_memory=False
        )

    def execute(self):
        return self.norm_layer(self.input_tensor)