import math

from core.base_processor import BaseProcessor
import random
import torch


class PositionalEncoding(BaseProcessor):
    def __init__(self):
        super().__init__()
        self.encoding = None
        self.seq_len = None
        self.d_model = None
        self.device = "cuda"

    def generate_config(self):
        # 序列长度与嵌入维度
        self.seq_len = random.randint(4, 512)
        self.d_model = random.choice([64, 128, 256, 512])

        self.config = {
            "seq_len": self.seq_len,
            "d_model": self.d_model
        }
        return self.config

    def setup(self):
        seq_len = self.config["seq_len"]
        d_model = self.config["d_model"]

        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)  # [seq_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))  # [d_model//2]
        pe = torch.zeros(seq_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)  # even index
        pe[:, 1::2] = torch.cos(position * div_term)  # odd index

        self.encoding = pe.unsqueeze(0).to(self.device)  # [1, seq_len, d_model]

    def execute(self):
        return self.encoding
