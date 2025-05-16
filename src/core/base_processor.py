import inspect
import pprint
from pathlib import Path
from abc import ABC, abstractmethod
from os.path import dirname
from utils.config_loader import load_config
from types import SimpleNamespace as SN

import torch


class BaseProcessor(ABC):
    """
        算子基本类，抽象方法
    """
    def __init__(self, args, logger):
        # 命令行名称参数
        self.args = args
        # 日志记录器
        self.logger = logger

        # 配置生成器的属性
        self.generator_config = SN(**self._load_config())

        # 默认有config属性
        self.config = None

        # 统一设备管理
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    def _load_config(self):
        """
            加载默认的配置
        """
        src_abs_path = Path(dirname(dirname(__file__)))
        # 获取实现类的模块的相对路径
        filepath = inspect.getfile(self.__class__)
        class_abs_path = Path(filepath)  # 提取文件名

        # 计算config文件路径
        relative_path = class_abs_path.relative_to(src_abs_path)  # 相对src的路径
        config_path = src_abs_path / "config" / relative_path.with_suffix(".yaml")

        default_config_path = src_abs_path / "config/default.yaml"

        config_path = str(config_path)
        default_config_path = str(default_config_path)

        final_config, config_in_mode, default_in_mode = load_config(config_path, default_config_path, self.args)

        self.logger.info(f"命令行参数为：\n{pprint.pformat(self.args, indent=4, width=1)}")
        self.logger.info(f"在该参数和配置文件下，{self.__class__.__name__}算子（模型）的配置生成模式为{final_config['mode']}")

        self.logger.info(f"{final_config['mode']}模式下{self.__class__.__name__}算子（模型）的生成测试配置的配置："
                         f"\n{pprint.pformat(config_in_mode, indent=4, width=1)}")
        self.logger.info(f"{final_config['mode']}模式下通用的生成测试配置的配置："
                         f"\n{pprint.pformat(default_in_mode, indent=4, width=1)}")
        self.logger.info(f"最终配置："
                         f"\n{pprint.pformat(final_config, indent=4, width=1)}")

        return final_config
