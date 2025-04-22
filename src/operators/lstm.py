from core.base_processor import BaseProcessor
import random
import torch
import torch.nn as nn


class LSTM(BaseProcessor):
    def __init__(self):
        super().__init__()
        self.input_tensor = None
        self.lstm = None

    def generate_config(self):
        # 参数上限定义
        MAX_INPUT_SIZE = 512
        MAX_HIDDEN_SIZE = 512
        MAX_LAYERS = 4
        MAX_SEQ_LEN = 100
        MAX_BATCH_SIZE = 64

        input_size = random.randint(1, MAX_INPUT_SIZE)
        hidden_size = random.randint(1, MAX_HIDDEN_SIZE)
        num_layers = random.randint(1, MAX_LAYERS)
        bidirectional = random.choice([True, False])
        # 更通用
        batch_first = True
        batch_size = random.randint(1, MAX_BATCH_SIZE)
        seq_len = random.randint(1, MAX_SEQ_LEN)

        self.config = {
            "input_size": input_size,
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "bidirectional": bidirectional,
            "batch_first": batch_first,
            "input_shape": [batch_size, seq_len, input_size],
        }
        return self.config

    def setup(self):
        input_size = self.config["input_size"]
        hidden_size = self.config["hidden_size"]
        num_layers = self.config["num_layers"]
        bidirectional = self.config["bidirectional"]
        batch_first = self.config["batch_first"]
        input_shape = self.config["input_shape"]
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=batch_first,
            bidirectional=bidirectional
        ).to("cuda")
        self.input_tensor = torch.randn(
            *input_shape,# "input_shape": [batch_size, seq_len, input_size],
            dtype=torch.float32,
            device="cuda"
        )

    def execute(self):
        # 也可以 return output, (hn, cn) 如果你想要状态也返回
        output, (hn, cn) = self.lstm(self.input_tensor)
        return output
