import torch
import torch.nn as nn
from .base_operation import BaseOperation
import random


class Conv2D(BaseOperation):
    def __init__(self):
        super().__init__()
        self.input_tensor = None
        self.output_tensor = None
        self.conv = None  # 卷积层

    def generate_config(self, C_in=3, K=3, H_out=112, W_out=112, C_out=64):
        """
        生成卷积层的配置参数：
        - 输入形状: (C_in, H_in, W_in) 从预定义列表中随机选择
        - 输出形状: (C_out, H_out, W_out)
        - 支持随机生成 kernel_size, stride, padding
        """
        # 预定义输入尺寸列表（高度和宽度）
        input_heights = [32, 64, 128, 256, 512]
        input_widths = [32, 64, 128, 256, 512]

        # 随机选择输入尺寸
        H_in = random.choice(input_heights)
        W_in = random.choice(input_widths)

        # 随机生成卷积参数
        kernel_sizes = [1, 3, 5, 7]
        strides = [1, 2, 3]
        paddings = [0, 1, 2, 3]

        # 随机选择参数并确保输出尺寸合法
        valid_params = False
        while not valid_params:
            kernel_size = random.choice(kernel_sizes)
            stride = random.choice(strides)
            padding = random.choice([p for p in paddings if p <= kernel_size // 2])

            # 计算理论输出尺寸
            calculated_H_out = (H_in + 2 * padding - kernel_size) // stride + 1
            calculated_W_out = (W_in + 2 * padding - kernel_size) // stride + 1

            # 检查是否匹配目标输出尺寸
            if calculated_H_out == H_out and calculated_W_out == W_out:
                valid_params = True

        # 随机选择输入输出通道数
        in_channels_list = [3, 16, 32, 64, 128, 256]
        out_channels_list = [64, 128, 256, 512]
        C_in = random.choice(in_channels_list)
        C_out = random.choice(out_channels_list)

        # 构建配置字典
        self.config = {
            "tensor_shape": (C_in, H_in, W_in),  # 输入形状 (C_in, H_in, W_in)
            "kernel_size": kernel_size,
            "stride": stride,
            "padding": padding,
            "out_channels": C_out
        }

        # 初始化卷积层
        self.conv = nn.Conv2d(
            in_channels=C_in,
            out_channels=C_out,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )

    def setup(self):
        """根据配置生成输入张量"""
        C_in, H_in, W_in = self.config["tensor_shape"]
        self.input_tensor = torch.randn(
            (C_in, H_in, W_in),
            dtype=torch.float32,
            device="cuda"
        )

    def execute(self):
        """执行卷积操作"""
        input_with_batch = self.input_tensor.unsqueeze(0)  # 添加 batch 维度
        output = self.conv(input_with_batch)  # 执行卷积
        self.output_tensor = output.squeeze(0)  # 移除 batch 维度
        return self.output_tensor


# 示例用法
# conv = Conv2D()
# conv.generate_config()  # 输入尺寸从列表中随机选取（如 [64, 128]）
# conv.setup()
# conv.execute()
#
# print("输入形状:", conv.input_tensor.shape)  # 例如: torch.Size([3, 64, 128])
# print("输出形状:", conv.output_tensor.shape)  # 例如: torch.Size([64, 112, 112])
