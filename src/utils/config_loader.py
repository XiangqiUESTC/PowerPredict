import collections
from copy import deepcopy

import yaml


def load_config(config_path, default_config_path, args):
    """
        这是一个用于加载算子（模型）yaml格式的生成配置的配置的方法

    """
    config_dict = {}
    default_config_dict = {}

    try:
        with open(config_path, 'r') as config_file:
            try:
                config_dict = yaml.safe_load(config_file)
            except yaml.YAMLError as exc:
                print(f"Can't load YAML from config file '{config_path}': {exc}")
    except FileNotFoundError:
        print(f"未找到配置文件：{config_path}")

    try:
        with open(default_config_path, 'r') as default_config_file:
            try:
                default_config_dict = yaml.safe_load(default_config_file)
            except yaml.YAMLError as exc:
                print(f"Can't load YAML from default config file '{default_config_path}': {exc}")
    except FileNotFoundError:
        print(f"未找到默认配置文件：{default_config_path}")

    # 深拷贝一份
    config_dict_copy = deepcopy(config_dict)
    default_config_dict_copy = deepcopy(default_config_dict)

    # 命令行参数优先
    config_dict_copy.update(args)

    # 最终的配置生成模式的决定顺序:命令行mode参数>算子（模型）的yaml文件mode配置>默认的配置文件的mode配置
    final_mode = config_dict_copy["mode"]

    # config是独属于具体的算子的模式配置
    config = config_dict_copy.get(final_mode, {})

    # default_config是所有算子通用的配置
    final_config = default_config_dict_copy.get(final_mode, {})

    # 用个性化的配置覆盖通用化的配置
    final_config.update(config)
    # 添加mode字段
    final_config["mode"] = final_mode

    return final_config, config_dict.get(final_mode, {}), default_config_dict.get(final_mode, {})
