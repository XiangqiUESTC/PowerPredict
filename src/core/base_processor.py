from abc import ABC, abstractmethod

import torch


class BaseProcessor(ABC):
    """
        算子基本类，抽象方法
    """
    def __init__(self):
        self.config = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 统一设备管理

    @abstractmethod
    def generate_config(self):
        """
            生成一个测试数据的配置
        """

    @abstractmethod
    def setup(self):
        """
            根据当前生成的配置字典的列表来装配数据
        """

    @abstractmethod
    def execute(self):
        """
            执行操作
        """
