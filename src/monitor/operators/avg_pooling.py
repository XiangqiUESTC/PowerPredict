import torch
import torch.nn as nn
from core.base_processor import BaseProcessor
import random


class AdaptiveAvgPool2D(BaseProcessor):
    def __init__(self, args, logger):
        super().__init__(args, logger)
        self.input_tensor = None
        self.output_tensor = None
        self.pool = None  # 自适应平均池化层
        self.in_channels = random.choice([3, 16, 32, 64, 128, 256])
        # 随机生成输入尺寸（H_in, W_in）
        self.input_heights = [32, 64, 128, 224, 256, 512, 768, 1024]
        self.input_widths = [32, 64, 128, 224, 256, 512, 768, 1024]
        self.H_in, self.W_in = random.choice(self.input_heights), random.choice(self.input_widths)
        self.times = 0
    def generate_config(self):
        """
        生成合法的自适应池化配置：
        - 随机生成输入尺寸（C_in, H_in, W_in）
        - 随机生成目标输出尺寸（H_out, W_out）
        """
        if self.mode == "random":
            self.in_channels = random.choice([3, 16, 32, 64, 128, 256])
            # 随机生成输入尺寸（H_in, W_in）
            self.input_heights = [32, 64, 128, 224, 256, 512, 768, 1024]
            self.input_widths = [32, 64, 128, 224,256, 512, 768, 1024]
            self.H_in, self.W_in = random.choice(self.input_heights), random.choice(self.input_widths)
            # 随机生成输出尺寸（确保合法）
            while True:
                self.H_out = random.choice([1, 2, 4, 8, 16, 32, 64])
                self.W_out = random.choice([1, 2, 4, 8, 16, 32, 64])
                # 确保输出尺寸不超过输入尺寸
                if self.H_out <= self.H_in and self.W_out <= self.W_in:
                    break

        elif self.mode == "increase":
            self.H_in = self.step_increment  * ((self.times + 1) // 2  + 1)
            self.W_in = self.step_increment  * ((self.times) // 2  + 1)
            self.times += 1
            while True:
                self.H_out = random.choice([1, 2, 4, 8, 16, 32, 64])
                self.W_out = random.choice([1, 2, 4, 8, 16, 32, 64])
                # 确保输出尺寸不超过输入尺寸
                if self.H_out <= self.H_in and self.W_out <= self.W_in:
                    break

        self.config = {
            "tensor_shape": (self.in_channels, self.H_in, self.W_in),
            "output_size": (self.H_out, self.W_out)  # 核心修改：直接指定目标输出尺寸
        }
        # 使用PyTorch原生自适应池化层
        self.pool = nn.AdaptiveAvgPool2d(output_size=(self.H_out, self.W_out))
    def setup(self):
        """根据配置生成输入张量"""
        C_in, H_in, W_in = self.config["tensor_shape"]
        self.input_tensor = torch.randn(
            (C_in, H_in, W_in),
            dtype=torch.float32,
            device=self.device
        )

    def execute(self):
        """执行自适应平均池化操作（无需手动调整参数）"""
        input_with_batch = self.input_tensor.unsqueeze(0)  # 添加 batch 维度
        output = self.pool(input_with_batch)  # PyTorch自动处理参数
        self.output_tensor = output.squeeze(0)  # 移除 batch 维度
        return self.output_tensor