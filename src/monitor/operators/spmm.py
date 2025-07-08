import math

import torch

from monitor.core.base_processor import BaseProcessor
import random


class Spmm(BaseProcessor):
    """
        稀疏矩阵乘法算子
    """
    def __init__(self, args, logger):
        super().__init__(args, logger)
        self.B = None
        self.A = None

        # 增加模式需要初始化一些变量
        if self.mode == "increase":
            self.times = 0

    def generate_config(self):
        # 每个维度的区间
        if self.mode == "random":
            single_dim_length_max = self.max_dim_size
            single_dim_length_min = self.min_dim_size
            # 稀疏度
            max_sparsity = self.max_sparsity
            min_sparsity = self.min_sparsity
            # 生成m×p和p×n的矩阵
            m = random.randint(single_dim_length_min, single_dim_length_max)
            p = random.randint(single_dim_length_min, single_dim_length_max)
            n = random.randint(single_dim_length_min, single_dim_length_max)
            # 随机生成非0元素个数
            nnz_min = math.ceil(m*p*min_sparsity)
            nnz_max = math.floor(m*p*max_sparsity)
            nnz = random.randint(nnz_min, nnz_max)
            # 当前的稀疏度
            sparsity = nnz/(m*p)
        elif self.mode == "increase":
            m = self.m + (self.times//3 + int(self.times % 3 >= 0)) * self.step_increment
            n = self.n + (self.times//3 + int(self.times % 3 >= 1)) * self.step_increment
            p = self.p + (self.times//3 + int(self.times % 3 >= 2)) * self.step_increment
            sparsity = round(min(self.start_sparsity + self.sparsity_increment * self.times, self.sparsity_max), 3)

            nnz = math.floor(m * p * sparsity)

            self.times += 1

        else:
            raise NotImplementedError

        self.config = {
            "monitors": m,
            "n": n,
            "p": p,
            "nnz": nnz,
            "sparsity": sparsity,
            "device": self.device,
        }

        return self.config

    def setup(self):
        m = self.config["monitors"]
        p = self.config["p"]
        n = self.config["n"]
        nnz = self.config["nnz"]
        # 构造稀疏矩阵的索引
        rows = torch.randint(0, m, (nnz,))
        cols = torch.randint(0, p, (nnz,))
        indices = torch.stack([rows, cols], dim=0)  # [2, nnz]
        # 构造对应的非零值
        values = torch.randn(nnz)
        # 构造稀疏矩阵 A，大小为 [monitors, p]
        self.A = torch.sparse_coo_tensor(indices, values, size=(m, p)).coalesce().to(self.device)
        # 构造稠密矩阵 B，大小为 [p, n]
        self.B = torch.randn(p, n).to(self.device)

    def execute(self):
        return torch.sparse.mm(self.A, self.B)

