from collections.abc import Mapping, Iterable
from copy import deepcopy
from types import SimpleNamespace

import yaml


def load_config(self_config_path, default_config_path, args):
    """
        这是一个用于加载算子（模型）yaml格式的生成配置的配置的方法
        self_config_path: 是自身的独立配置
        default_config_path: 是通用的配置
        args: 是命令行参数
    """
    self_config_dict = {}
    default_config_dict = {}

    # 尝试读取独立配置yaml文件
    try:
        with open(self_config_path, 'r') as config_file:
            try:
                self_config_dict = yaml.safe_load(config_file)
            except yaml.YAMLError as exc:
                print(f"Can't load YAML from config file '{self_config_path}': {exc}")
    except FileNotFoundError:
        print(f"未找到配置文件：{self_config_path}")

    # 尝试读取通用配置yaml文件
    try:
        with open(default_config_path, 'r') as default_config_file:
            try:
                default_config_dict = yaml.safe_load(default_config_file)
            except yaml.YAMLError as exc:
                print(f"Can't load YAML from default config file '{default_config_path}': {exc}")
    except FileNotFoundError:
        print(f"未找到默认配置文件：{default_config_path}")

    # 深拷贝一份
    self_config_dict_copy = deepcopy(self_config_dict)
    default_config_dict_copy = deepcopy(default_config_dict)

    # 命令行参数优先
    self_config_dict_copy.update(args)

    # 最终的配置生成模式的决定顺序:命令行mode参数>算子（模型）的yaml文件mode配置>默认的配置文件的mode配置
    final_mode = self_config_dict_copy["mode"]
    # config是独属于具体的算子的模式配置
    config = self_config_dict_copy.get(final_mode, {})
    # default_config是所有算子通用的配置
    final_config = default_config_dict_copy.get(final_mode, {})
    if final_config is None:
        final_config = {}

    # 用个性化的配置覆盖通用化的配置
    if config:
        final_config.update(config)
    # 添加一般字段
    for key, value in default_config_dict_copy.items():
        if not isinstance(value, dict):
            final_config[key] = value
    # 添加一般字段
    for key, value in self_config_dict_copy.items():
        if not isinstance(value, dict):
            final_config[key] = value

    final_config["mode"] = final_mode

    return final_config, self_config_dict.get(final_mode, {}), default_config_dict.get(final_mode, {})


def dict_to_attribute_recursively(config):
    """
        递归地将字典（包括多层嵌套）转换为 SimpleNamespace 对象

        参数:
            config: 输入配置，可以是字典、列表、元组或其他类型

        返回:
            转换后的 SimpleNamespace 对象或原始类型
        """
    # 如果是字典类型，递归处理每个值并创建命名空间
    if isinstance(config, Mapping):
        for key, value in config.items():
            config[key] = dict_to_attribute_recursively(value)
        return SimpleNamespace(**config)

    # 如果是列表或元组，递归处理每个元素
    elif isinstance(config, Iterable) and not isinstance(config, str):
        processed = []
        for item in config:
            processed.append(dict_to_attribute_recursively(item))
        return type(config)(processed)  # 保持原始容器类型

    # 其他类型直接返回
    else:
        return config