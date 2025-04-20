import torch.nn.functional as F
import torch
from .base_operation import BaseOperation
import random


class MaxPool2D(BaseOperation):
    def __init__(self):
        super().__init__()
        self.input_tensor = None
        self.output_tensor = None

    def generate_config(self):
        """
        随机生成最大池化层的合法参数组合：
        - kernel_size ∈ [1, 2, 3, 5, 7, 9]
        - stride ∈ [1, 2, 3, 4, 5]
        - padding ∈ [0, 1, 2, 3, 4]（需 ≤ kernel_size//2）
        - 输入输出尺寸保证输出为整数且 >0
        """
        # 随机选择池化参数
        kernel_sizes = [1, 2, 3, 5, 7, 9]
        strides = [1, 2, 3, 4, 5]
        paddings = [0, 1, 2, 3, 4]

        # 随机选择池化类型（当前仅支持 max）
        pool_type = 'max'

        # 随机选择输入输出通道数
        in_channels_list = [3, 16, 32, 64, 128, 256]
        out_channels_list = [64, 128, 256, 512]

        # 随机生成输入输出尺寸（H_in, W_in, H_out, W_out）
        heights = [32, 64, 128, 256, 512, 768, 1024]
        widths = [32, 64, 128, 256, 512, 768, 1024]

        while True:  # 循环直到生成合法参数
            # 随机选择参数
            kernel_size = random.choice(kernel_sizes)
            stride = random.choice(strides)
            padding = random.choice([p for p in paddings if p <= kernel_size // 2])  # 确保 padding 合法
            in_channels = random.choice(in_channels_list)
            out_channels = random.choice(out_channels_list)
            H_in, W_in = random.sample(heights + widths, 2)  # 随机选择输入尺寸
            H_out, W_out = random.sample(heights + widths, 2)  # 随机选择输出尺寸

            # 计算实际输出尺寸（验证合法性）
            try:
                actual_H_out = (H_in + 2 * padding - kernel_size) // stride + 1
                actual_W_out = (W_in + 2 * padding - kernel_size) // stride + 1
                if actual_H_out == H_out and actual_W_out == W_out and actual_H_out > 0 and actual_W_out > 0:
                    break
            except:
                continue

        # 构建配置字典
        self.config = {
            "tensor_shape": (in_channels, H_in, W_in),  # 输入形状 (C_in, H_in, W_in)
            "kernel_size": kernel_size,
            "stride": stride,
            "padding": padding,
            "out_channels": out_channels  # 输出通道数
        }

    def setup(self):
        """根据配置生成输入张量"""
        C_in, H_in, W_in = self.config["tensor_shape"]
        self.input_tensor = torch.randn(
            (C_in, H_in, W_in),
            dtype=torch.float32,
            device="cuda"
        )

    def execute(self):
        """执行最大池化操作"""
        input_with_batch = self.input_tensor.unsqueeze(0)  # 添加 batch 维度
        output = F.max_pool2d(
            input=input_with_batch,
            kernel_size=self.config["kernel_size"],
            stride=self.config["stride"],
            padding=self.config["padding"]
        )
        self.output_tensor = output.squeeze(0)  # 移除 batch 维度
        return self.output_tensor


# 示例用法
# max_pool = MaxPool2D()
# max_pool.generate_config()  # 生成随机参数（如 kernel=3, stride=2, padding=1）
# max_pool.setup()
# max_pool.execute()
#
# print("输入形状:", max_pool.input_tensor.shape)  # 例如: torch.Size([64, 112, 112])
# print("输出形状:", max_pool.output_tensor.shape)  # 例如: torch.Size([64, 56, 56])
