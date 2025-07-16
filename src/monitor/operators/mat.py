import torch
from core.base_processor import BaseProcessor
import random


class Mat(BaseProcessor):
    def __init__(self, args, logger):
        super().__init__(args, logger)
        self.mat1 = None
        self.mat2 = None  # 相当于另一个矩阵
        self.output_tensor = None
        self.times = 0
        self.input_dim = [random.randint(1, 1048), random.randint(1, 1048)]
        self.weight_dim = [self.input_dim[1], random.randint(1, 1048)]
        self.output_dim = [self.input_dim[0], self.weight_dim[1]]
    def generate_config(self):
        """
        生成矩阵乘法参数：
        - 输入形状: (M, K)
        - 权重形状: (K, N)
        - 输出形状: (M, N)
        """
        # 随机生成矩阵维度（支持自定义默认参数）

        if self.mode == "increase":
            self.input_dim[0]= self.BASIC_NUMBER  * ((self.times + 2) // 3  + 1)
            self.input_dim[1]= self.BASIC_NUMBER  * ((self.times + 1) // 3  + 1)
            self.weight_dim[0] = self.BASIC_NUMBER * ((self.times + 1) // 3 + 1)
            self.weight_dim[1] = self.BASIC_NUMBER * ((self.times) // 3 + 1)
            self.times += 1
        else:
            self.input_dim = [random.randint(1, 1048), random.randint(1, 1048)]
            self.weight_dim = [input_dim[1], random.randint(1, 1048)]
            self.output_dim = [input_dim[0], weight_dim[1]]
        self.output_dim = [self.input_dim[0], self.weight_dim[1]]
        self.config = {
            "mat1_shape": self.input_dim,
            "mat2_shape": self.weight_dim,
            "output_shape": self.output_dim
        }
    def setup(self):
        """生成输入张量"""
        M, K = self.config["mat1_shape"]
        K, N = self.config["mat2_shape"]
        self.mat1 = torch.randn(
            M, K,
            dtype=torch.float32,
            device=self.device
        )
        # 初始化矩阵2
        self.mat2 = torch.randn(
            K, N,
            dtype=torch.float32,
            device=self.device
        )

    def execute(self):
        """执行矩阵乘法"""
        # 矩阵乘法：input @ weight
        self.output_tensor = torch.matmul(self.mat1, self.mat2)
        return self.output_tensor
