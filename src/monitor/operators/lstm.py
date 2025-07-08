from monitor.core.base_processor import BaseProcessor
import random
import torch
import torch.nn as nn


class LSTM(BaseProcessor):
    def __init__(self, args, logger):
        super().__init__(args, logger)
        self.input_tensor = None
        self.lstm = None

        # 设置默认参数（可以从 args 读取或直接写死）
        self.min_input_size = 1
        self.max_input_size = 1024
        self.min_hidden_size = 1
        self.max_hidden_size = 1024
        self.min_layers = 1
        self.max_layers = 4
        self.min_seq_len = 4
        self.max_seq_len = 512
        self.min_batch_size = 1
        self.max_batch_size = 64
        self.bidirectional_options = [True, False]

        # 递增模式用初始参数
        if self.mode == "increase":
            self.times = 0  # 增长计数器
            self.base_input_size = 16
            self.base_hidden_size = 16
            self.base_num_layers = 1
            self.base_batch_size = 2
            self.base_seq_len = 8
            self.step_increment = 16

            # 递增模式固定参数
            self.bidirectional = False
            self.batch_first = True

            # 设置上限
            self.max_input_size = 4096
            self.max_hidden_size = 4096
            self.max_num_layers = 16
            self.max_seq_len = 2048
            self.max_batch_size = 256

    def generate_config(self):
        if self.mode == "random":
            input_size = random.randint(self.min_input_size, self.max_input_size)
            hidden_size = random.randint(self.min_hidden_size, self.max_hidden_size)
            num_layers = random.randint(self.min_layers, self.max_layers)
            bidirectional = random.choice(self.bidirectional_options)
            batch_first = random.choice([True, False])
            batch_size = random.randint(self.min_batch_size, self.max_batch_size)
            seq_len = random.randint(self.min_seq_len, self.max_seq_len)

        elif self.mode == "increase":
            input_size = min(self.base_input_size + self.times * self.step_increment, self.max_input_size)
            hidden_size = min(self.base_hidden_size + self.times * self.step_increment, self.max_hidden_size)
            num_layers = min(self.base_num_layers + self.times // 2, self.max_num_layers)
            batch_size = min(self.base_batch_size + self.times * 2, self.max_batch_size)
            seq_len = min(self.base_seq_len + self.times * 4, self.max_seq_len)
            bidirectional = self.bidirectional
            batch_first = self.batch_first
            self.times += 1

        else:
            raise NotImplementedError(f"Unsupported mode: {self.mode}")

        self.config = {
            "input_size": input_size,
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "bidirectional": bidirectional,
            "batch_first": batch_first,
            "input_shape": [batch_size, seq_len, input_size] if batch_first else [seq_len, batch_size, input_size],
            "device": self.device,
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
        ).to(self.device)

        self.input_tensor = torch.randn(
            input_shape,
            dtype=torch.float32,
            device=self.device
        )

    def execute(self):
        output, (hn, cn) = self.lstm(self.input_tensor)
        return output
