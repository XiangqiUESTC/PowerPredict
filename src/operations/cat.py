from .base_operation import BaseOperation
import torch
import random


class Cat(BaseOperation):
    def __init__(self):
        super().__init__()
        self.tensors = []

    def generate_config(self):
        # 连接的tensor的个数
        MAX_TENSOR_NUM = 4
        MIN_TENSOR_NUM = 2
        # 最大维数
        MAX_DIM_NUM = 4
        # 最小维数
        MIN_DIM_NUM = 1
        # 每个维度的区间
        SINGLE_DIM_LENGTH_MAX = 512
        SINGLE_DIM_LENGTH_MIN = 1

        # 随机生成tensor数量
        tensor_num = random.randint(MIN_TENSOR_NUM, MAX_TENSOR_NUM)

        # 随机维度数量
        k = random.randint(MIN_DIM_NUM, MAX_DIM_NUM)
        # 生成维度值
        arr = [random.randint(SINGLE_DIM_LENGTH_MIN, SINGLE_DIM_LENGTH_MAX) for _ in range(k)]
        # copy tensor_num行
        arr_2d = [[dim for dim in arr]
                  for _ in range(tensor_num)]
        # 随机选择操作的维度
        dim = random.randint(0, k - 1)
        # 重写要拼接的维度的大小
        for i in range(tensor_num):
            arr_2d[i][dim] = random.randint(SINGLE_DIM_LENGTH_MIN, SINGLE_DIM_LENGTH_MAX)

        # 除了连接
        self.config = {
            "tensor_shapes": arr_2d,
            "dim": dim,
        }
        return self.config

    def setup(self):
        # 从配置中拿到基本数据
        tensor_shapes = self.config['tensor_shapes']

        # 生成多个tensor
        for tensor_shape in tensor_shapes:
            self.tensors.append(torch.randn(
                tensor_shape,
                dtype=torch.float,
                device="cuda",
                pin_memory=False)
            )

    def execute(self):
        dim = self.config['dim']
        return torch.cat(self.tensors, dim=dim)
