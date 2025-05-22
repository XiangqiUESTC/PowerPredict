from core.base_processor import BaseProcessor
import random
import torch


class Add(BaseProcessor):
    def __init__(self, args, logger):
        super().__init__(args, logger)
        self.A = None
        self.B = None
        if self.mode == "increase":
            self.times = 0
            self.dim = self.DIM_NUM
            self.arr = []
            self.current_index = 0
            for i in range(self.dim):
                self.arr.append(self.step_increment)#初始化

    def generate_config(self):
        # 最大维数
        MAX_DIM_NUM = 4
        # 最小维数
        MIN_DIM_NUM = 1
        # 每个维度的区间
        SINGLE_DIM_LENGTH_MAX = 256
        SINGLE_DIM_LENGTH_MIN = 1

        # 随机维度数量
        k = random.randint(MIN_DIM_NUM, MAX_DIM_NUM)
        # 生成维度值
        arr = [random.randint(SINGLE_DIM_LENGTH_MIN, SINGLE_DIM_LENGTH_MAX) for _ in range(k)]
        # 生成配置字典
        self.config = {
            "tensor_shape": arr
        }
        # return self.config
        if self.mode == "random":
           pass
        elif self.generator_config.mode == "increase":
            self.arr[self.current_index] += self.step_increment
            self.current_index = (self.current_index + 1) % self.dim
            self.config = {
                "tensor_shape":self.arr
            }
        else:
            raise NotImplementedError
    def setup(self):
        tensor_shape = self.config["tensor_shape"]
        self.A = torch.randn(
            tensor_shape,
            dtype=torch.float,
            device=self.device,
            pin_memory=False
        )
        self.B = torch.randn(
            tensor_shape,
            dtype=torch.float,
            device=self.device,
            pin_memory=False
        )

    def execute(self):
        return torch.add(self.A, self.B)
