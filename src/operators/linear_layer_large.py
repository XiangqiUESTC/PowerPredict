import random

import torch
import torch.nn as nn
from core.base_processor import BaseProcessor


class LinearLayerLarge(BaseProcessor):
    def __init__(self):
        super().__init__()
        self.input_tensor = None
        self.output_tensor = None
        self.linear = None  # 全连接层

    def generate_config(self, C_in_max=150000, C_out_max=150000):
        """
        生成全连接层的配置参数：
        - 输入形状: (C_in, 1, 1)（假设输入已被展平）
        - 输出形状: (C_out)
        """
        C_in = random.randint(1,C_in_max)
        C_out = random.randint(1,C_out_max)
        self.config = {
            "tensor_shape": (C_in, 1, 1),  # 输入形状 (C_in, 1, 1)
            "out_channels": C_out          # 输出通道数（即 C_out）
        }
        # 初始化全连接层
        self.linear = nn.Linear(C_in, C_out).to(self.device)

    def setup(self):
        """根据配置生成输入张量"""
        C_in, H_in, W_in = self.config["tensor_shape"]
        self.input_tensor = torch.randn(
            (C_in, 1, 1),  # 输入形状 (25088, 1, 1)
            dtype=torch.float32,
            device=self.device
        )

    def execute(self):
        """执行全连接操作"""
        # 添加 batch 维度 → (1, C_in, 1, 1)
        input_with_batch = self.input_tensor.unsqueeze(0)
        # 展平输入 → (1, C_in)
        flattened = input_with_batch.view(input_with_batch.size(0), -1)
        # 全连接变换 → (1, C_out)
        output = self.linear(flattened)
        # 移除 batch 维度 → (C_out,)
        self.output_tensor = output.squeeze(0)
        return self.output_tensor

