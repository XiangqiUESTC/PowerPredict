from torch import nn

from .base_operation import BaseOperation
import random
import torch


class Embedding(BaseOperation):
    def __init__(self):
        super().__init__()
        self.embedding = None
        self.input_tensor = None

    def generate_config(self):
        # 最大词表大小和向量维度
        MAX_NUM_EMBEDDINGS = 10000
        MAX_EMBED_DIM = 512
        MAX_BATCH_SIZE = 64
        MAX_SEQ_LEN = 128

        # 词表大小
        num_embeddings = random.randint(10, MAX_NUM_EMBEDDINGS)
        # 每个词向量的维度
        embedding_dim = random.randint(4, MAX_EMBED_DIM)
        batch_size = random.randint(1, MAX_BATCH_SIZE)
        seq_len = random.randint(1, MAX_SEQ_LEN)
        # 生成配置
        self.config = {
            "num_embeddings": num_embeddings,
            "embedding_dim": embedding_dim,
            "input_shape": [batch_size, seq_len],  # 模拟输入句子的 ID
        }
        return self.config

    def setup(self):
        num_embeddings = self.config["num_embeddings"]
        embedding_dim = self.config["embedding_dim"]
        input_shape = self.config["input_shape"]

        self.embedding = nn.Embedding(num_embeddings, embedding_dim).to("cuda")
        self.input_tensor = torch.randint(
            low=0,
            high=num_embeddings,
            size=tuple(input_shape),
            device="cuda",
            dtype=torch.long
        )

    def execute(self):
        return self.embedding(self.input_tensor)
