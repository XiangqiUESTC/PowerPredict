import sys
from copy import deepcopy
from os.path import dirname, join, abspath
import threading
import pynvml
from datetime import datetime

from operators import OPERATOR_REGISTRY
from models import MODEL_REGISTRY
from utils.logger import Logger
from monitor.monitor import run_all_and_monitor
from thirdparty.monitor_hardware import monitor_main


# ----------------- 主函数 -----------------
if __name__ == '__main__':
    # 初始化监测工具pynvml
    pynvml.nvmlInit()
    # 初始化日志器
    log_dir = join(abspath(dirname(dirname(abspath(__file__)))), "log")
    logger = Logger(log_dir)

    # 解析命令行中的名称参数
    argv = deepcopy(sys.argv)
    args = {}
    # 解析所有--key=value的命令行参数，并将其从命令行参数中删掉，保存在字典arg_dict中
    index_to_del = []
    for i, arg in enumerate(argv[1:]):
        if arg.startswith("--"):
            splits = arg[2:].split("=")
            key = splits[0]
            value = splits[1]
            args[key] = value
            # 注意这里是i+1,因为截取的时候从第二个参数截取的
            index_to_del.append(i + 1)

    # 从后往前删，避免出错
    index_to_del.reverse()
    for i in index_to_del:
        del argv[i]


    # 解析剩余的命令，获取必要的参数
    op_names = None
    num_samples = None

    if len(argv) < 3:
        logger.error("Usage:python main.py [op_1,op_2,...] test_rounds")
        logger.error("未能启动实验，请提供算子或模型名及测试次数！")
    else:
        # 解析剩余的参数
        # 首先检查有没有未实现的算子和模型名
        # 同时注册算子和模型
        REGISTRY = {
            **OPERATOR_REGISTRY,
            **MODEL_REGISTRY,
        }
        op_names = argv[1].split(",")
        not_implements = [op_name for op_name in op_names if op_name not in REGISTRY]
        if len(not_implements) > 0:
            logger.error(f"找不到算子或模型{not_implements}，您确定在operator或models模块下实现"
                         f"并在OPERATOR_REGISTRY或MODEL_REGISTRY中注册了它吗？")
            # 如果有就退出程序
            exit(-1)
        try:
            num_samples = int(argv[2])
        except ValueError as e:
            logger.error(f"argument 2必须是个整数！而不是{argv[2]}")
            # 如果第二个参数不是整数就退出
            exit(-1)
        # 如果有多余的参数,进行警告,仍然运行
        if len(argv)>3:
            logger.warning(f"忽略多余参数:{argv[3:]}")


    logger.info(f"命令行参数解析完成，开始实验，测试的算子或模型有：\n{op_names}")
    logger.info(f"每个算子或模型测试{num_samples}次")

    # 生成一下结果文件夹
    t = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    test_folder =  f"test-{t}" if not hasattr(args, "test_name") else f"{args['test_name']}-{t}"
    results_folder = join(dirname(dirname(abspath(__file__))), "results", test_folder)

    # 根据参数决定是否运行第三方监测程序
    monitor_flag = {}
    monitor_thread = None
    if "third_monitor" in args and args["third_monitor"] == "true":
        monitor_flag = {
            "flag": True
        }
        monitor_thread = threading.Thread(target=monitor_main, args=(logger, monitor_flag))
        monitor_thread.start()

    # 运行并监测所有算子的消耗情况
    run_all_and_monitor(args, logger, op_names, num_samples, results_folder)

    if "third_monitor" in args and args["third_monitor"] == "true":
        # 结束第三方监测进程
        monitor_flag["flag"] = False
        monitor_thread.join()

    exit(1)