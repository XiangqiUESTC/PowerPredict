import torch
from torch import nn

from core.base_processor import BaseProcessor
import random


class MaxPool2D(BaseProcessor):
    def __init__(self, args, logger):
        super().__init__(args, logger)
        self.pool = None
        self.input_tensor = None
        self.output_tensor = None

        self.in_channels = random.choice([3, 16, 32, 64, 128, 256])
        # 随机生成输入尺寸（H_in, W_in）
        self.input_heights = [32, 64, 128, 224, 256, 512, 768, 1024]
        self.input_widths = [32, 64, 128, 224, 256, 512, 768, 1024]
        self.H_in, self.W_in = random.choice(self.input_heights), random.choice(self.input_widths)
        self.times = 0
        self.in_channels_list = [3, 16, 32, 64, 128, 256]
        self.in_channels = random.choice(self.in_channels_list)
        self.out_channels = self.in_channels
    def generate_config(self):
        """
            生成随机池化层的合法参数组合：
            - 输入尺寸 (C_in, H_in, W_in)
            - 池化参数：kernel_size, stride, padding（符合实际运算规则）
            - 池化类型：MaxPool2d
            """
        # 随机选择输入输出通道数（允许不同）

        if self.mode == "random":
            # 随机生成输入尺寸（H_in, W_in）
            heights = [32, 64, 128, 256, 512, 768, 1024]
            widths = [32, 64, 128, 256, 512, 768, 1024]
            self.H_in, self.W_in = random.sample(heights + widths, 2)
            # 生成合法池化参数（确保输出尺寸 ≥1）
            while True:
                kernel_size = random.choice([1, 2, 3, 4])
                max_padding = kernel_size // 2
                padding = random.randint(0, max_padding)
                stride = random.randint(1, kernel_size + 2 * padding)

                # 计算输出尺寸
                self.H_out = (self.H_in + 2 * padding - kernel_size) // stride + 1
                self.W_out = (self.W_in + 2 * padding - kernel_size) // stride + 1
                # 验证输出尺寸有效性
                if self.H_out >= 1 and self.W_out >= 1:
                    break
            self.config = {
                "tensor_shape": (self.in_channels, self.H_in, self.W_in),
                "kernel_size": kernel_size,
                "stride": stride,
                "padding": padding,
                "output_size": (self.out_channels, self.H_out, self.W_out)
            }

            self.pool = nn.MaxPool2d(kernel_size=kernel_size,
                                     stride=stride,
                                     padding=padding)
        else:
            self.H_in = self.step_increment * ((self.times + 1) // 2 + 1)
            self.W_in = self.step_increment * ((self.times) // 2 + 1)
            self.times += 1
            while True:
                kernel_size = random.choice([2])
                padding = kernel_size // 2
                stride = 2

                # 计算输出尺寸
                self.H_out = (self.H_in + 2 * padding - kernel_size) // stride + 1
                self.W_out = (self.W_in + 2 * padding - kernel_size) // stride + 1
                # 验证输出尺寸有效性
                if self.H_out >= 1 and self.W_out >= 1:
                    break
            self.config = {
                "tensor_shape": (self.in_channels, self.H_in, self.W_in),
                "kernel_size": kernel_size,
                "stride": stride,
                "padding": padding,
                "output_size": (self.out_channels, self.H_out, self.W_out)
            }

            self.pool = nn.MaxPool2d(kernel_size=kernel_size,
                                     stride=stride,
                                     padding=padding)

    def setup(self):
        """根据配置生成输入张量"""
        C_in, H_in, W_in = self.config["tensor_shape"]
        self.input_tensor = torch.randn(
            (C_in, H_in, W_in),
            dtype=torch.float32,
            device=self.device
        )

    def execute(self):
        """执行最大池化操作"""
        input_with_batch = self.input_tensor.unsqueeze(0)  # 添加 batch 维度
        output = self.pool (
            input_with_batch,
        )
        self.output_tensor = output.squeeze(0)  # 移除 batch 维度
        return self.output_tensor

