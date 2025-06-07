import math
from core.base_processor import BaseProcessor
import random
import torch


class PositionalEncoding(BaseProcessor):
    def __init__(self, args, logger):
        super().__init__(args, logger)
        self.encoding = None
        self.seq_len = None
        self.d_model = None

        # 递增模式初始化
        if self.mode == "increase":
            self.times = 0  # 运行次数计数器
            # 设置递增模式上限
            self.max_seq_len = 2048
            self.max_d_model = 2048

    def generate_config(self):
        if self.mode == "random":
            # 使用yaml配置参数
            seq_len = random.randint(self.min_seq_len, self.max_seq_len)
            d_model = random.choice(self.d_model_options)

        elif self.mode == "increase":
            # 线性递增参数
            seq_len = self.base_seq_len + self.times * self.step_increment
            d_model = self.base_d_model + self.times * self.step_increment

            # 应用上限保护
            seq_len = min(seq_len, self.max_seq_len)
            d_model = min(d_model, self.max_d_model)

            self.times += 1  # 更新计数器

        else:
            raise NotImplementedError(f"Unsupported mode: {self.mode}")

        # 确保d_model是偶数（因为计算需要奇偶索引）
        if d_model % 2 != 0:
            d_model += 1

        self.config = {
            "seq_len": seq_len,
            "d_model": d_model
        }
        return self.config

    def setup(self):
        seq_len = self.config["seq_len"]
        d_model = self.config["d_model"]

        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1).to(self.device)  # [seq_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2).float().to(self.device) *
                             (-math.log(10000.0) / d_model))  # [d_model//2]
        pe = torch.zeros(seq_len, d_model).to(self.device)

        # 计算位置编码
        pe[:, 0::2] = torch.sin(position * div_term)  # even index
        pe[:, 1::2] = torch.cos(position * div_term)  # odd index

        self.encoding = pe.unsqueeze(0)  # [1, seq_len, d_model]

    def execute(self):
        return self.encoding