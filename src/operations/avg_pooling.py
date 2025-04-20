import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_operation import BaseOperation
import random


class AdaptiveAvgPool2D(BaseOperation):
    def __init__(self):
        super().__init__()
        self.input_tensor = None
        self.output_tensor = None
        self.pool = None  # 自适应平均池化层

    def generate_config(self):
        """
        生成自适应平均池化层的合法参数组合：
        - 输入尺寸 (C_in, H_in, W_in)
        - 输出尺寸 (H_out, W_out) ≤ 输入尺寸
        - 通道数保持不变（C_in = C_out）
        """
        # 随机选择输入输出通道数（确保一致）
        in_channels_list = [3, 16, 32, 64, 128, 256]
        in_channels = random.choice(in_channels_list)
        out_channels = in_channels  # 通道数不变

        # 随机生成输入尺寸（H_in, W_in）
        heights = [32, 64, 128, 256, 512, 768, 1024]
        widths = [32, 64, 128, 256, 512, 768, 1024]
        H_in, W_in = random.sample(heights + widths, 2)

        # 生成输出尺寸（H_out ≤ H_in, W_out ≤ W_in）
        H_out = random.choice([h for h in heights if h <= H_in])
        W_out = random.choice([w for w in widths if w <= W_in])

        self.config = {
            "tensor_shape": (in_channels, H_in, W_in),  # 输入形状 (C_in, H_in, W_in)
            "output_size": (H_out, W_out),              # 输出尺寸 (H_out, W_out)
            "out_channels": out_channels                # 输出通道数（等于输入通道数）
        }

        # 初始化自适应平均池化层
        self.pool = nn.AdaptiveAvgPool2d(self.config["output_size"])

    def setup(self):
        """根据配置生成输入张量"""
        C_in, H_in, W_in = self.config["tensor_shape"]
        self.input_tensor = torch.randn(
            (C_in, H_in, W_in),
            dtype=torch.float32,
            device="cuda"
        )

    def execute(self):
        """执行自适应平均池化操作"""
        input_with_batch = self.input_tensor.unsqueeze(0)  # 添加 batch 维度
        output = self.pool(input_with_batch)               # 自动调整池化窗口大小和步长
        self.output_tensor = output.squeeze(0)             # 移除 batch 维度
        return self.output_tensor


# 示例用法
# adaptive_avg_pool = AdaptiveAvgPool2D()
# adaptive_avg_pool.generate_config()  # 生成配置（如输入 [64, 112, 112]，输出 [64, 56, 56]）
# adaptive_avg_pool.setup()
# adaptive_avg_pool.execute()
#
# print("输入形状:", adaptive_avg_pool.input_tensor.shape)  # 例如: torch.Size([64, 112, 112])
# print("输出形状:", adaptive_avg_pool.output_tensor.shape)  # 例如: torch.Size([64, 56, 56])
