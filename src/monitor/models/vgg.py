import torchvision
import random
from monitor.core.base_processor import BaseProcessor
import torch.nn as nn
from monitor.dataset.dataset import *


class Vgg(BaseProcessor):
    def __init__(self, args, logger, num_classes=1000):
        super().__init__(args, logger)
        self.num_classes = num_classes
        self.model = None
        self.input_tensor = None
        self.output_tensor = None
        self.config = {}
        self.adaptive_pool_size = 7  # 自适应池化目标尺寸

        self.data = None
        self.data_loader = None
        self.dataset_name = 'cifar10'

    def generate_config(self):
        """生成支持224x224的尺寸"""
        vgg_versions = ['vgg11', 'vgg13', 'vgg16', 'vgg19']
        BATCH_SIZE_MAX = 32
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
            "model_version": random.choice(vgg_versions),
            "pretrained": random.choice([True, False]),
            "num_classes": self.num_classes
        })

    def setup(self):
        """生成支持224x224的尺寸的初始化模型"""
        batch_size = self.config["batch_size"]
        # 加载数据集
        self.data_loader = load_dataset(self.dataset_name, batch_size)
        # 权重映射表（更新为使用DEFAULT权重）
        weight_map = {
            'vgg11': torchvision.models.VGG11_Weights.DEFAULT,
            'vgg13': torchvision.models.VGG13_Weights.DEFAULT,
            'vgg16': torchvision.models.VGG16_Weights.DEFAULT,
            'vgg19': torchvision.models.VGG19_Weights.DEFAULT
        }

        try:
            # 初始化模型
            constructor = getattr(torchvision.models, self.config["model_version"])
            weights = weight_map[self.config["model_version"]] if self.config["pretrained"] else None
            self.model = constructor(weights=weights)
            # 动态调整分类层
            if self.num_classes != 1000:
                # 自动计算输入特征维度
                with torch.no_grad():
                    test_input = torch.randn(1, 3, *self.config["input_shape"][2:])
                    features = self.model.features(test_input)
                    in_features = features.view(-1).shape[0]

                # 重构分类器
                self.model.classifier = nn.Sequential(
                    nn.Linear(in_features, 4096),
                    nn.ReLU(True),
                    nn.Dropout(p=0.5),
                    nn.Linear(4096, 4096),
                    nn.ReLU(True),
                    nn.Dropout(p=0.5),
                    nn.Linear(4096, self.num_classes)
                )

            # 设备转移
            self.model = self.model.to(self.device).eval()
            # 取出一组数据即可
            self.data, _ = next(iter(self.data_loader))
            self.data = self.data.to(self.device)

        except KeyError as e:
            raise ValueError(f"不支持的模型版本: {self.config['model_version']}") from e
        except AttributeError as e:
            raise ImportError(f"当前torchvision版本不支持该模型: {str(e)}") from e

    def execute(self):
        # 确保评估模式
        self.model.eval()
        return self.model(self.data)
