import math

import torch

from .base_operation import BaseOperation
import random


class Spmm(BaseOperation):
    """
        稀疏矩阵乘法算子
    """
    def __init__(self):
        super().__init__()
        self.B = None
        self.A = None

    def generate_config(self):
        # 每个维度的区间
        SINGLE_DIM_LENGTH_MAX = 512
        SINGLE_DIM_LENGTH_MIN = 1
        # 稀疏度
        MAX_SPARSITY = 0.3
        MIN_SPARSITY = 0.001
        # 生成m×p和p×n的矩阵
        m = random.randint(SINGLE_DIM_LENGTH_MIN, SINGLE_DIM_LENGTH_MAX)
        p = random.randint(SINGLE_DIM_LENGTH_MIN, SINGLE_DIM_LENGTH_MAX)
        n = random.randint(SINGLE_DIM_LENGTH_MIN, SINGLE_DIM_LENGTH_MAX)
        # 随机生成非0元素个数
        nnz_min = math.ceil(m*p*MIN_SPARSITY)
        nnz_max = math.floor(m*p*MAX_SPARSITY)
        nnz = random.randint(nnz_min, nnz_max)
        # 当前的稀疏度
        sparsity = nnz/(m*p)

        self.config = {
            "m": m,
            "p": p,
            "n": n,
            "nnz": nnz,
            "sparsity": sparsity
        }
        return self.config

    def setup(self):
        m = self.config["m"]
        p = self.config["p"]
        n = self.config["n"]
        nnz = self.config["nnz"]

        # 构造稀疏矩阵的索引
        rows = torch.randint(0, m, (nnz,))
        cols = torch.randint(0, p, (nnz,))
        indices = torch.stack([rows, cols], dim=0)  # [2, nnz]

        # 构造对应的非零值
        values = torch.randn(nnz)

        # 构造稀疏矩阵 A，大小为 [m, p]
        self.A = torch.sparse_coo_tensor(indices, values, size=(m, p)).coalesce()

        # 构造稠密矩阵 B，大小为 [p, n]
        self.B = torch.randn(p, n)

    def execute(self):
        return torch.sparse.mm(self.A, self.B)

