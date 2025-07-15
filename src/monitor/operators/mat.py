import torch
from monitor.core.base_processor import BaseProcessor
import random


class Mat(BaseProcessor):
    def __init__(self, args, logger):
        super().__init__(args, logger)
        self.mat1 = None
        self.mat2 = None  # 相当于另一个矩阵
        self.output_tensor = None

        # 增加模式需要初始化一些变量
        if self.mode == "increase":
            self.times = 0

    def generate_config(self):
        """
        生成矩阵乘法参数：
        - 输入形状: (M, K)
        - 权重形状: (K, N)
        - 输出形状: (M, N)
        """
        m = None
        n = None
        p = None

        if self.mode == "increase":
            # 随机生成矩阵维度（支持自定义默认参数）
            m = self.m + (self.times//3 + int(self.times % 3 >= 0)) * self.step_increment
            n = self.n + (self.times//3 + int(self.times % 3 >= 1)) * self.step_increment
            p = self.p + (self.times//3 + int(self.times % 3 >= 2)) * self.step_increment

            self.times += 1

        self.config = {
            "m": m,
            "p": n,
            "n": p,
        }

    def setup(self):
        """生成输入张量"""
        m = self.config["m"]
        p = self.config["p"]
        n = self.config["n"]
        self.mat1 = torch.randn(
            m, p,
            dtype=torch.float32,
            device=self.device
        )
        # 初始化矩阵2
        self.mat2 = torch.randn(
            p, n,
            dtype=torch.float32,
            device=self.device
        )

    def execute(self):
        """执行矩阵乘法"""
        # 矩阵乘法：input @ weight
        self.output_tensor = torch.matmul(self.mat1, self.mat2)
        return self.output_tensor
