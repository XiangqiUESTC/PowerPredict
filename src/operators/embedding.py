from torch import nn
from core.base_processor import BaseProcessor
import random
import torch


class Embedding(BaseProcessor):
    def __init__(self, args, logger):
        super().__init__(args, logger)
        self.embedding = None
        self.input_tensor = None

        # 递增模式初始化
        if self.mode == "increase":
            self.times = 0  # 运行次数计数器
            # 在递增模式中设置独立的上限参数
            self.max_num_embeddings = 20000
            self.max_embedding_dim = 1024
            self.max_batch_size = 256
            self.max_seq_len = 512

    def generate_config(self):
        if self.mode == "random":
            # 使用yaml配置参数
            num_embeddings = random.randint(
                self.min_num_embeddings,
                self.max_num_embeddings
            )
            embedding_dim = random.randint(
                self.min_embedding_dim,
                self.max_embedding_dim
            )
            batch_size = random.randint(
                self.min_batch_size,
                self.max_batch_size
            )
            seq_len = random.randint(
                self.min_seq_len,
                self.max_seq_len
            )

        elif self.mode == "increase":
            # 线性递增所有参数
            num_embeddings = self.base_num_embeddings + self.times * self.step_increment
            embedding_dim = self.base_embedding_dim + self.times * self.step_increment
            batch_size = self.base_batch_size + self.times * self.step_increment
            seq_len = self.base_seq_len + self.times * self.step_increment

            # 确保不超过递增模式的最大值
            num_embeddings = min(num_embeddings, self.max_num_embeddings)
            embedding_dim = min(embedding_dim, self.max_embedding_dim)
            batch_size = min(batch_size, self.max_batch_size)
            seq_len = min(seq_len, self.max_seq_len)

            self.times += 1  # 更新计数器

        else:
            raise NotImplementedError(f"Unsupported mode: {self.mode}")

        self.config = {
            "num_embeddings": num_embeddings,
            "embedding_dim": embedding_dim,
            "input_shape": [batch_size, seq_len],
            "device": self.device,
        }
        return self.config

    def setup(self):
        num_embeddings = self.config["num_embeddings"]
        embedding_dim = self.config["embedding_dim"]
        input_shape = self.config["input_shape"]

        # 创建并移动Embedding层到指定设备
        self.embedding = nn.Embedding(num_embeddings, embedding_dim).to(self.device)

        # 创建输入张量并移动到指定设备
        self.input_tensor = torch.randint(
            low=0,
            high=num_embeddings,
            size=tuple(input_shape),
            device=self.device,
            dtype=torch.long
        )

    def execute(self):
        return self.embedding(self.input_tensor)