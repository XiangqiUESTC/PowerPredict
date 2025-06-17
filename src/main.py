import sys
from copy import deepcopy
from os.path import dirname, join, abspath
import time
import threading
import subprocess
import csv
from datetime import datetime
from utils.device import is_device_avail_on_torch

import torch

from operators import *
from models import *
import os
from utils.logger import Logger
from utils.csv_utils import write_csv


def get_gpu_info(device, l):
    """
        :param l: 日志器
        :param device: 不同的设备类型
        :return: gpu数据，包括功率、utilization、memory
    """
    if device == 'cpu':
        return True
    elif device == "cuda":
        try:
            output = subprocess.check_output([
                'nvidia-smi',
                '--query-gpu=power.draw,utilization.gpu,memory.used',
                '--format=csv,noheader,nounits'
            ])
            info = output.decode('utf-8').strip()
            return info
        except Exception as e:
            l.error(f"获取{device}设备信息出错")
            l.exception(e)
            return None
    elif device == "npu":
        try:
            # 执行 npu-smi info 命令获取原始输出
            output = subprocess.check_output("npu-smi info", shell=True, text=True)

            # 解析输出，提取所有NPU的功耗值
            power_values = []
            for line in output.split('\n'):
                # print("start get_gpu_power")
                # print("-" * 50)
                # print("-" * 50)
                # print(line)
                # print("-" * 50)
                # print("-" * 50)
                if '| 0     ' in line:  # 匹配NPU信息行
                    parts = [p.strip() for p in line.split('|') if p.strip()]
                    # print("parts :", parts)
                    # if len(parts) >= 4:
                    if parts[1] == 'OK':
                        power = parts[2].split()[0]  # 提取功耗值(如"93.9")
                        power_values.append(float(power))
                        # print("gpu功耗: ", power)

            # 计算平均功耗(如果有多个NPU)
            if power_values:
                return sum(power_values) / len(power_values)
            return None
        except Exception as e:
            print(f"获取NPU功耗出错: {e}")
            return None
    elif device == "xpu":
        return None
    else:
        raise Exception("Unknown device")

# ----------------- GPU 监控线程 -----------------
def gpu_monitor_thread_func(logfile, stop_flag, l, device):
    """
    ARGS:
        logfile: gpu监测信息的输出文件的绝对路径
        stop_flag: stop_flag是一个字典，字典属于非基本变量，通过字典里面的值来在外部控制进程的结束
        l: 日志实例对象
        device: gpu设备类型
    DESCRIPTION:
        gpu监控线程函数
    """
    # 新增目录创建逻辑
    os.makedirs(dirname(logfile), exist_ok=True)
    # GPU监测代码
    with open(logfile, 'w') as f:
        f.write("timestamp, power.draw[W], util[%], memory[MiB]\n")
        while not stop_flag["stop"]:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            info = get_gpu_info(device, l)
            if info:
                f.write(f"{timestamp}, {info}\n")

def operation_monitor(operation, operation_name, l, num_sample=1, loop_per_sample=64, preheat=80):
    """
    ARGS:
        operation: operation是基本的算子或者模型,同时是base_operation的实现类的实例对象
        operation_name: 算子的名称,用于命名最后的文件
        l: 日志实例对象
        num_sample=1: 该算子要测试的默认数据组数
        loop_per_sample=64: 每个算子重复测试的次数,最后取平均
        preheat=80: 预热次数

    DESCRIPTION:
        创建两个线程，分别监控该算子在运算时的CPU和GPU数据，分析并合并输出为csv
    """
    file_name = operation_name + ".csv"

    # 文件日期
    file_date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # csv中间文件夹
    temp_dir = join(abspath(dirname(dirname(abspath(__file__)))), "temp")
    # csv结果文件夹
    result_folder = operation.test_name + "__" + file_date
    result_dir = join(abspath(dirname(dirname(abspath(__file__)))), "results", result_folder)
    # 最终结果
    records = {}

    # 获取当前的设备设置信息
    device = op.device
    device_avail = is_device_avail_on_torch(device)

    # 如果设备正常，运行测试
    if device_avail:
        # 循环num_sample次
        for j in range(num_sample):
            try:
                # 开始测试
                l.info(f"Test Case {j + 1} for {op_name}: Starting monitoring and computation...")
                # 生成此次测试的配置
                _ = operation.generate_config()
                # 装配数据
                try:
                    operation.setup()
                except Exception as error:
                    l.exception(error)
                    continue

                # 预热GPU
                try:
                    for i_ in range(preheat):
                        # 创建两个随机矩阵（尺寸可根据需要调整）
                        a = torch.randn(100, 100).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))  # 100x100矩阵
                        b = torch.randn(100, 100).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
                        # 执行矩阵乘法 → 触发CUDA核函数
                        c = torch.matmul(a, b)  # 结果矩阵 100x100

                except Exception as error:
                    l.error(f"预热失败: {str(error)}")
                    torch.cuda.empty_cache()  # 显存异常时清空缓存
                    continue

                # 启动GPU监控线程(此处的gpu可能是cuda，也可能是xpu、npu等)
                gpu_log = join(temp_dir, f'gpu_{operation_name}_{file_date}.csv')
                stop_gpu = {"stop": False}
                gpu_thread = threading.Thread(target=gpu_monitor_thread_func, args=(gpu_log, stop_gpu, l, device))
                gpu_thread.start()

                # 记录时间戳
                start_time = datetime.now().isoformat()
                # 记录持续时间（毫秒）
                start_time_ns = time.time_ns()
                # 重复执行，不断采样
                f = False
                for _ in range(loop_per_sample):
                    try:
                        operation.execute()
                    except Exception as error:
                        l.error(error)
                        l.exception(error)
                        f = True
                        torch.cuda.empty_cache()  # 释放缓存
                        # 跳出
                        break
                if f:
                    # 此次error
                    continue
                # 记录时间戳
                end_time_ns = time.time_ns()
                # 记录持续时间（毫秒）
                end_time = datetime.now().isoformat()

                # 结束监控线程
                stop_gpu["stop"] = True
                gpu_thread.join()

                # 保证采样完整
                time.sleep(12)

                # 解析GPU功耗数据
                powers = []
                utils = []
                memory_used = []
                with open(gpu_log, 'r') as f:
                    # 跳过文件头
                    next(f)
                    for data_line in f:
                        data_item = data_line.strip().split(',')
                        if len(data_item) == 4:
                            powers.append(float(data_item[1]))
                            utils.append(float(data_item[2]))
                            memory_used.append(float(data_item[3]))
                # 取平均,保留两位小数
                avg_power = round(sum(powers) / len(powers), 2) if powers else 0
                max_power = round(max(powers, default=0), 2)

                # 取平均,保留两位小数
                avg_utils = round(sum(utils) / len(utils), 2) if utils else 0
                max_utils = round(max(utils, default=0), 2)

                # 取平均,保留两位小数
                avg_memory_used = round(sum(memory_used) / len(memory_used), 2) if memory_used else 0
                max_memory_used = round(max(memory_used, default=0), 2)

                gpu_data = {
                    "max_gpu_power": max_power,
                    "avg_gpu_power": avg_power,
                    "max_gpu_util": max_utils,
                    "avg_gpu_utils": avg_utils,
                    "max_gpu_memory_used": max_memory_used,
                    "avg_gpu_memory_used": avg_memory_used,
                }

                # 计算时间
                duration = round((end_time_ns - start_time_ns) / loop_per_sample, 2)

                other_data = {
                    "duration": duration,
                    "start_time": start_time,
                    "end_time": end_time,
                }

                test_config = op.config

                l.info(f"监测到数据\n{test_config}{other_data}{gpu_data}")

                # 解析数据字典
                dictionaries = [test_config, other_data, gpu_data]
                for dictionary in dictionaries:
                    for k, v in dictionary.items():
                        # 第一次的时候需要初始化
                        if k not in records:
                            records[k] = []
                        records[k].append(v)

            except Exception as error:
                logger.error(f"{operation_name}第{j + 1}/{loop_per_sample}次重复测试失败，原因是：\n")
                logger.exception(error)
        # 开始写最终的数据
        result_file = join(result_dir, file_name)
        # 写入CSV
        write_csv(result_file, records)
    else:
        l.error(f"{device}不可用,已跳过测试")




# ----------------- 主函数 -----------------
if __name__ == '__main__':
    # 同时注册算子和模型
    REGISTRY = {
        **OPERATOR_REGISTRY,
        **MODEL_REGISTRY,
    }

    # 默认测试所有
    op_names = [
        *REGISTRY.keys()
    ]

    num_samples = 3

    # 初始化日志
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
            index_to_del.append(i + 1)

    index_to_del.reverse()

    for i in index_to_del:
        del argv[i]

    # 解析命令
    if len(argv) < 2:
        logger.warning("没有提供要测试的算子名称和测试次数！将运行默认测试用例!")
    elif len(argv) < 3:
        logger.warning(f"没有指定测试次数！默认每个项目测试{num_samples}次")
    else:
        op_names = argv[1].split(",")
        not_implements = [op_name for op_name in op_names if op_name not in REGISTRY]
        if len(not_implements) > 0:
            logger.error(f"找不到算子或模型{not_implements}，您确定在operator或models模块下实现"
                         f"并在OPERATOR_REGISTRY或MODEL_REGISTRY中注册了它吗？")
            exit(-1)
        try:
            num_samples = int(argv[2])
        except ValueError as e:
            logger.error(f"参数2必须是个整数！而不是{argv[2]}")
            exit(-1)

    logger.info(f"命令行参数解析完成，开始实验，测试的算子或模型有：\n{op_names}")
    logger.info(f"每个算子或模型测试{num_samples}次")

    # 开始主循环
    for op_name in op_names:
        op = REGISTRY[op_name](args, logger)
        logger.info(f"对{op_name}的实验开始!测试{num_samples}次!")
        logger.info(f"默认设备是{op.device}")
        operation_monitor(
            op,
            op_name,
            logger,
            num_samples,
        )
        logger.info(f"对算子{op_name}的{num_samples}次测试结束!")
        logger.info(
            "--------------------------------------------------------------------------------------------------------------------------------------")  #分割

    logger.info("实验结束！")
