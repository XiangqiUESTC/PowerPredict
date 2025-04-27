import torch
import torch.nn as nn
import random
from core.base_processor import BaseProcessor


class Conv2DLarge(BaseProcessor):
    def __init__(self):
        super().__init__()
        self.input_tensor = None
        self.output_tensor = None
        self.conv = None  # 卷积层
        self.iterNumber = 1
        self.basicNumber = 128
    def generate_config(self):
        """
        生成随机卷积层的合法参数组合：
        - 输入形状: (C_in, H_in, W_in)
        - 卷积参数: kernel_size, stride, padding
        - 输出形状: (C_out, H_out, W_out)
        """
        # 随机选择输入输出通道数
        # in_channels_list = [256, 512, 768, 1024]
        in_channels_list = [256]

        C_in = random.choice(in_channels_list)
        # out_channels_list = [256, 512, 768, 1024]
        out_channels_list = [256]
        C_out = random.choice(out_channels_list)

        # 随机生成输入尺寸 (H_in, W_in)
        # heights = [256, 512,1024, 2048, 3072, 4096]
        # widths = [256, 512,1024, 2048, 3072, 4096]
        self.iterNumber += 1
        H_in = self.basicNumber  * ((self.iterNumber + 1) // 2 )
        W_in = self.basicNumber  * (self.iterNumber // 2 )

        # H_in, W_in = random.sample(heights + widths, 2)

        # 生成合法卷积参数（确保输出尺寸 ≥1）
        while True:
            kernel_size = random.choice([1, 3,5, 7, 9, 11])
            stride = random.choice([1, 2, 3])
            max_padding = kernel_size // 2
            padding = random.randint(0, max_padding)

            # 计算理论输出尺寸
            H_out = (H_in + 2 * padding - kernel_size) // stride + 1
            W_out = (W_in + 2 * padding - kernel_size) // stride + 1

            # 验证输出尺寸有效性
            if H_out >= 1 and W_out >= 1:
                break
        self.config = {
            "tensor_shape": (C_in, H_in, W_in),  # 输入形状 (C_in, H_in, W_in)
            "kernel_size": kernel_size,
            "stride": stride,
            "padding": padding,
            "out_channels": C_out,
            "output_size": (C_out, H_out, W_out)
        }

        # 初始化卷积层
        self.conv = nn.Conv2d(
            in_channels=C_in,#输入通道
            out_channels=C_out,#输出通道
            kernel_size=kernel_size,#卷积核尺寸
            stride=stride,#步长
            padding=padding#
        )

    def setup(self):
        """根据配置生成输入张量并移动模型到GPU"""
        C_in, H_in, W_in = self.config["tensor_shape"]
        self.input_tensor = torch.randn(
            (C_in, H_in, W_in),
            dtype=torch.float32,
            device=self.device
        )
        # 将卷积层移动到GPU
        self.conv = self.conv.to(self.device)

    def execute(self):
        """执行卷积操作"""
        input_with_batch = self.input_tensor.unsqueeze(0)  # 添加 batch 维度
        output = self.conv(input_with_batch)               # 执行卷积
        self.output_tensor = output.squeeze(0)             # 移除 batch 维度
        return self.output_tensor
