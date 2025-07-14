import random
from monitor.core.base_processor import BaseProcessor
import torch.nn as nn
import torchvision.models as models
from torchvision.models import AlexNet_Weights  # 新增导入
from monitor.dataset.dataset import *


class AlexNet(BaseProcessor):
    def __init__(self, args, logger, num_classes=1000):
        super().__init__(args, logger)
        self.model = None
        self.data = None
        self.data_loader = None
        self.num_classes = num_classes
        self.config = {}
        self.dataset_name = 'cifar10'

    def generate_config(self):
        # 批次上下限
        BATCH_SIZE_MAX = 512
        BATCH_SIZE_MIN = 1
        # 选择数据大小批次
        batch_size = random.randint(BATCH_SIZE_MIN, BATCH_SIZE_MAX)
        self.config["batch_size"] = batch_size
        # 从可用数据集中随机选择一个数据集
        dataset = ["cifar10", "flowers102"]

        self.dataset_name = random.choice(dataset)
        self.num_classes = DATASET_INFO[self.dataset_name]['num_classes']

        # 更新输入形状为合法格式
        self.config.update({
            "input_shape": [1, 3, 224, 224],  # 修正为CHW格式
            "pretrained": random.choice([True, False]),
            "num_classes": self.num_classes
        })

    def setup(self):
        batch_size = self.config["batch_size"]
        # 加载数据集
        self.data_loader = load_dataset(self.dataset_name, batch_size)

        # 初始化模型（使用新版权重API）
        weights = AlexNet_Weights.IMAGENET1K_V1 if self.config["pretrained"] else None
        self.model = models.alexnet(weights=weights)

        # 修改分类层
        if self.num_classes != 1000:
            self.model.classifier[6] = nn.Linear(4096, self.num_classes)

        # 设备转移和模式设置
        self.model = self.model.to(self.device).eval()  # 直接使用self.model

        # 取出一组数据即可
        self.data, _ = next(iter(self.data_loader))
        self.data = self.data.to(self.device)

    def execute(self):
        # 确保评估模式
        self.model.eval()
        return self.model(self.data)
