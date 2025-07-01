import sys
from copy import deepcopy
from os.path import dirname, join, abspath
import time
import threading
import subprocess
from datetime import datetime
from utils.device import is_device_avail_on_torch

import torch

from operators import *
from models import *
import os
from utils.logger import Logger
from utils.csv_utils import write_csv
from thirdparty.monitor_hardware import monitor_main
from monitor.gpu import get_gpu_model


def get_gpu_info(device, l):
    """
        :param l: 日志器
        :param device: 不同的设备类型
        :return: gpu数据，包括功率、utilization、memory
    """
    if not isinstance(device, str):
        exc = TypeError("设备应该是一个字符串！")
        l.exception(exc)
        raise exc

    if device == 'cpu':
        return True
    elif device.startswith("cuda"):
        # 默认使用第一个设备
        device_num = 0
        # 从device字符串中提取设备编号
        if ':' in device:
            try:
                # 提取冒号后的数字部分
                device_num = int(device.split(':')[1])
            except (ValueError, IndexError):
                # 处理无效的设备编号格式
                print(f"警告: 无效的设备格式 '{device}'，默认使用设备0")
                device_num = 0
        try:
            output = subprocess.check_output([
                'nvidia-smi',
                '--query-gpu=power.draw,utilization.gpu,memory.used',
                '--format=csv,noheader,nounits'
            ])
            info = output.decode('utf-8').strip().splitlines()
            return info[device_num]
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


# ----------------- 主函数 -----------------
if __name__ == '__main__':
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

    # 运行第三方监测程序
    monitor_flag = {
        "flag": True
    }
    monitor_thread = threading.Thread(target=monitor_main, args=(logger, monitor_flag))
    monitor_thread.start()




    # 结束控制进程
    monitor_flag["flag"] = False
    monitor_thread.join()

    exit(1)