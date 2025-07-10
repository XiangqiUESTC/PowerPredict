import pprint
import sys
from copy import deepcopy
import yaml
from winsound import SND_NOWAIT

from utils.logger import Logger
from os.path import join, abspath, dirname
from types import SimpleNamespace

from train.trainer import TRAINER_REGISTRY
from train.predictor import PREDICTOR_REGISTRY
from train.preprocessor import PREPROCESSOR_REGISTRY

from utils.csv_utils import  merge_csv_to_pd

if __name__ == '__main__':
    # 命令示例
    # python cat cat_results --arg1=xxx --arg2=xxx
    # 获取项目根目录
    project_abs_path = dirname(dirname(abspath(__file__)))

    # 初始化日志器
    log_dir = join(project_abs_path, "log", "train")
    logger = Logger(log_dir)


    # 解析参数
    argv = deepcopy(sys.argv)
    ## 分离名称参数和位置参数
    index_to_del = []
    pos_arg_key = []
    pos_arg_value = []
    for i, arg in enumerate(argv[1:]):
        if arg.startswith("--"):
            splits = arg[2:].split("=")
            try:
                key = splits[0]
                value = splits[1]
                pos_arg_key.append(key)
                pos_arg_value.append(value)
            except IndexError:
                logger.error(f"位置参数{arg}未采取--arg_name=arg_value的格式！")
                exit(-1)
            # 注意这里是i+1,因为截取的时候从第二个参数截取的
            index_to_del.append(i + 1)

    ## 从原参数数组中删除名称参数
    index_to_del.reverse()
    for i in index_to_del:
        del argv[i]

    pos_args = {}
    ## 处理名称参数，主要是解决嵌套问题
    assert len(pos_arg_key) == len(pos_arg_value)
    for (raw_key, value,) in zip(pos_arg_key, pos_arg_value):
        keys = raw_key.split(".")
        dic = pos_args
        for key, i in zip(keys, range(len(keys))):
            # 若不是最后一层
            if i != len(keys)-1:
                if key not in dic:
                    dic[key] = {}
                dic = dic[key]
            else:
                dic[key] = value

    ## 处理位置参数
    ### 检查合法性
    if len(argv) != 3:
        logger.info(f"位置参数有：{argv}")
        if len(argv) < 3:
            logger.error("未提供足够的参数！")
        elif len(argv) > 3:
            logger.error("检查是否有多余的参数！")
        logger.error("正确的训练指令是: python train.py [训练任务名] [训练数据文件夹] [--pos_arg_name_1=value_1 ...]")
        exit(-1)

    task_name = argv[1]
    pos_args["raw_folder"] = argv[2]

    # 读取并按优先级合并配置
    ##  配置包括默认配置、训练任务配置和命令行配置
    default_config = None
    task_config = None
    ### 读取默认配置
    try:
        default_yaml_f = open("train/config/default.yaml", mode="r", encoding="utf-8")
        try:
            default_config = yaml.safe_load(default_yaml_f)
        except yaml.YAMLError as exc:
            logger.error("读取default.yaml失败，原因是：")
            logger.exception(exc)
            default_config = {}
    except FileNotFoundError:
        logger.warning("没有找到src/train/config/default.yaml文件！默认配置为空！")
        default_config = {}
    ### 读取任务配置
    try:
        task_yaml_f = open(f"train/config/tasks/{task_name}.yaml", mode="r", encoding="utf-8")
        try:
            task_config = yaml.safe_load(task_yaml_f)
        except yaml.YAMLError as exc:
            logger.error("读取default.yaml失败，原因是：")
            logger.exception(exc)
            task_config = {}
    except FileNotFoundError:
        logger.warning(f"没有找到train/config/tasks/{task_name}.yaml文件！任务配置为空！")
        task_config = {}

    logger.info(f"默认配置为：\n{pprint.pformat(default_config, indent=4, width=1)}")
    logger.info(f"任务{task_name}对应的配置为：\n{pprint.pformat(task_config, indent=4, width=1)}")
    logger.info(f"命令行配置为：\n{pprint.pformat(pos_args, indent=4, width=1)}")

    # config是最终的配置
    config = {}
    config.update(default_config)
    config.update(task_config)
    config.update(pos_args)
    logger.info(f"最终配置为：\n{pprint.pformat(config, indent=4, width=1)}")
    config = SimpleNamespace(**config)

    # 读取数据
    # 获取原始数据文件夹
    abs_raw_folder = join(project_abs_path, config.data_root_folder, config.source_data_folder, config.raw_folder)
    print(abs_raw_folder)

    raw_data = merge_csv_to_pd(config.raw_file_regex, abs_raw_folder)
    print(raw_data)


    # 开始训练
    preprocessor = PREPROCESSOR_REGISTRY[config.preprocessor]()
    predictor = PREDICTOR_REGISTRY[config.predictor]()
    trainer = TRAINER_REGISTRY[config.trainer]()

