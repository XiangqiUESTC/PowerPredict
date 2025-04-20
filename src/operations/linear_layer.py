import torch
import torch.nn as nn
from .base_operation import BaseOperation


class LinearLayer(BaseOperation):
    def __init__(self):
        super().__init__()
        self.input_tensor = None
        self.output_tensor = None
        self.linear = None  # 全连接层

    def generate_config(self, C_in=25088, C_out=4096):
        """
        生成全连接层的配置参数：
        - 输入形状: (C_in, 1, 1)（假设输入已被展平）
        - 输出形状: (C_out)
        """
        self.config = {
            "tensor_shape": (C_in, 1, 1),  # 输入形状 (C_in, 1, 1)
            "out_channels": C_out          # 输出通道数（即 C_out）
        }
        # 初始化全连接层
        self.linear = nn.Linear(C_in, C_out)

    def setup(self):
        """根据配置生成输入张量"""
        C_in, H_in, W_in = self.config["tensor_shape"]
        self.input_tensor = torch.randn(
            (C_in, H_in, W_in),  # 输入形状 (25088, 1, 1)
            dtype=torch.float32,
            device="cuda"
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


# 示例用法
# linear_layer = LinearLayer()
# linear_layer.generate_config()  # 默认参数 C_in=25088, C_out=4096
# linear_layer.setup()
# linear_layer.execute()
#
# print("输入形状:", linear_layer.input_tensor.shape)  # torch.Size([25088, 1, 1])
# print("输出形状:", linear_layer.output_tensor.shape)  # torch.Size([4096])
