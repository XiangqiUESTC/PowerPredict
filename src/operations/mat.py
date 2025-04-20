import torch
from .base_operation import BaseOperation
import random


class Mat(BaseOperation):
    def __init__(self):
        super().__init__()
        self.input_tensor = None
        self.weight = None  # 相当于另一个矩阵
        self.output_tensor = None

    def generate_config(self, M=32, N=64, K=128):
        """
        生成矩阵乘法参数：
        - 输入形状: (M, K)
        - 权重形状: (K, N)
        - 输出形状: (M, N)
        """
        # 随机生成矩阵维度（支持自定义默认参数）
        input_dim = (random.randint(1, M), random.randint(1, K))
        weight_dim = (input_dim[1], random.randint(1, N))
        output_dim = (input_dim[0], weight_dim[1])

        self.config = {
            "input_shape": input_dim,
            "weight_shape": weight_dim,
            "output_shape": output_dim
        }

        # 初始化矩阵参数
        self.weight = torch.randn(
            weight_dim[0], weight_dim[1],
            dtype=torch.float32,
            device="cuda"
        )

    def setup(self):
        """生成输入张量"""
        M, K = self.config["input_shape"]
        self.input_tensor = torch.randn(
            M, K,
            dtype=torch.float32,
            device="cuda"
        )

    def execute(self):
        """执行矩阵乘法"""
        # 矩阵乘法：input @ weight
        self.output_tensor = torch.matmul(self.input_tensor, self.weight)
        return self.output_tensor


# 示例用法
# mm = Mat()
# mm.generate_config()  # 默认生成 (M=32, K=128) x (K=128, N=64) 的矩阵乘法
# mm.setup()
# mm.execute()
#
# print("输入形状:", mm.input_tensor.shape)  # 例如: torch.Size([32, 128])
# print("权重形状:", mm.weight.shape)  # 例如: torch.Size([128, 64])
# print("输出形状:", mm.output_tensor.shape)  # torch.Size([32, 64])
