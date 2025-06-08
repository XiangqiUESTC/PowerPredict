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
            self.dim = random.randint(self.MIN_DIM_NUM, self.MAX_DIM_NUM)
            self.arr = []
            self.current_index = 0
            for i in range(self.dim):
                self.arr.append(self.step_increment)#初始化

    def generate_config(self):
        # 最大维数

        # return self.config
        MAX_DIM_NUM = self.MAX_DIM_NUM
        # 最小维数
        MIN_DIM_NUM = self.MIN_DIM_NUM
        # 每个维度的区间
        SINGLE_DIM_LENGTH_MAX = self.SINGLE_DIM_LENGTH_MAX
        SINGLE_DIM_LENGTH_MIN = self.SINGLE_DIM_LENGTH_MIN
        # 随机维度数量

        if self.mode == "random":
            k = random.randint(MIN_DIM_NUM, MAX_DIM_NUM)
            # 生成维度值
            arr = [random.randint(SINGLE_DIM_LENGTH_MIN, SINGLE_DIM_LENGTH_MAX) for _ in range(k)]
            # 生成配置字典
            self.config = {
                "tensor_shape": arr,
                "dim": k
            }
        elif self.mode == "increase":
            self.arr[self.current_index] += self.step_increment
            if self.arr[self.current_index] > SINGLE_DIM_LENGTH_MAX:
               self.arr[self.current_index] = SINGLE_DIM_LENGTH_MAX
            self.current_index = (self.current_index + 1) % self.dim
            self.config = {
                "tensor_shape":self.arr,
                "dim": self.dim
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
