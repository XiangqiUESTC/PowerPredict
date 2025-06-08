from copy import deepcopy

from core.base_processor import BaseProcessor
import torch
import random


class Cat(BaseProcessor):
    def __init__(self, args, logger):
        super().__init__(args, logger)

        self.logger = logger
        self.tensors = []

        # 增加模式需要初始化一些变量
        if self.mode == "increase":
            self.times = 0
            self.cat_dim_sizes = [random.randint(self.min_dim_size, self.max_dim_size) for _ in
                                  range(self.tensor_num)]
            self.base_shapes = [random.randint(self.min_dim_size, self.max_dim_size) for _ in range(self.dim_num)]

            self.cat_dim = random.randint(0, self.dim_num)

    def generate_config(self):
        if self.mode == "random":
            # 读一下配置
            max_tensor_num = self.max_tensor_num
            min_tensor_num = self.min_tensor_num
            max_tensor_dim = self.max_tensor_dim
            min_tensor_dim = self.min_tensor_dim
            max_dim_size = self.max_dim_size
            min_dim_size = self.min_dim_size

            # 随机生成连接的tensor数量
            tensor_num = random.randint(min_tensor_num, max_tensor_num)

            # 随机生成基础维度数
            base_dims = random.randint(min_tensor_dim, max_tensor_dim)

            # 生成基础形状（所有张量共享的基础维度）
            base_shape = [
                random.randint(min_dim_size, max_dim_size) for _ in range(base_dims)
            ]

            # 随机选择要拼接的维度
            concat_dim = random.randint(0, base_dims - 1)

            # 生成每个张量的形状（仅修改拼接维度）
            tensor_shapes = []
            for _ in range(tensor_num):
                new_shape = base_shape.copy()
                # 确保拼接维度至少保留1个元素
                new_shape[concat_dim] = random.randint(min_dim_size, max_dim_size)
                tensor_shapes.append(new_shape)

            self.config = {
                "tensor_shapes": tensor_shapes,
                "dim": concat_dim,
            }

        elif self.mode == "increase":
            tensor_shapes = [deepcopy(self.base_shapes) for _ in range(self.tensor_num)]
            for i, shape in enumerate(tensor_shapes):
                shape[self.cat_dim] = self.cat_dim_sizes[i]

            # 递增
            for i in range(self.dim_num):
                self.base_shapes[i] += self.step_increment

            self.config = {
                "tensor_shapes": tensor_shapes,
                "dim": self.cat_dim,
            }

        else:
            raise NotImplementedError

        return self.config

    def setup(self):
        """根据配置生成输入张量"""
        tensor_shapes = self.config['tensor_shapes']
        # 初始化tensors为空列表
        self.tensors = []

        for shape in tensor_shapes:
            self.tensors.append(torch.randn(
                shape,
                dtype=torch.float32,
                device=self.device
            ))

    def execute(self):
        """执行张量拼接"""
        # 验证所有张量维度数一致
        tensor_dims = [t.dim() for t in self.tensors]
        if len(set(tensor_dims)) != 1:
            raise RuntimeError(f"张量维度不一致: {tensor_dims}")
        # 验证拼接维度有效性
        concat_dim = self.config['dim']
        if concat_dim >= tensor_dims[0]:
            raise RuntimeError(f"无效的拼接维度 {concat_dim} (张量是 {tensor_dims[0]} 维)")
        return torch.cat(self.tensors, dim=concat_dim)
