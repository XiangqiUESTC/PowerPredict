import sys
from os.path import dirname, join, abspath
import time
import threading
import subprocess
import csv
from datetime import datetime

import torch

from operators import *
from models import *
import os
from utils.logger import Logger
from utils.csv_utils import write_csv


def operation_monitor(operation, operation_name, l, num_sample=1, loop_per_sample=64, preheat=8):
    """
    ARGS:
        operation: operation是基本的算子,同时是base_operation的实现类的实例对象
        operation_name: 算子的名称,用于命名最后的文件
        num_sample: 该算子要测试的数据组数
        loop_per_sample: 每个算子重复测试的次数,最后取平均
        preheat: 预热次数
        l: 日志实例对象
    DESCRIPTION:
        创建两个线程，分别监控该算子在运算时的CPU和GPU数据，分析并合并输出为csv
    """
    file_name = operation_name + ".csv"
    # with open(file_name, 'w', newline='') as f:
    #     writer = csv.writer(f)
    #     writer.writerow([
    #     'MatrixSize', 'Duration(s)', 'AvgCPUPkgW', 'AvgGPUPower(W)', 'AvgGPUUtil(%)', 'AvgGPUMem(MiB)'
    #     ])

    # csv中间文件夹
    temp_dir = join(abspath(dirname(dirname(abspath(__file__)))), "temp")
    # csv结果文件夹
    result_dir = join(abspath(dirname(dirname(abspath(__file__)))), "results")
    # 文件日期
    file_date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # 最终结果
    records = {}
    # 循环num_sample次
    for i in range(num_sample):
        # 开始测试
        l.info(f"Test Case {i + 1} for {op_name}: Starting monitoring and computation...")
        # 生成此次测试的配置
        _ = operation.generate_config()
        # 装配数据
        try:
            operation.setup()
        except Exception as error:
            l.error(error)


        # 预热GPU
        try:
            # 创建两个随机矩阵（尺寸可根据需要调整）
            a = torch.randn(100, 100).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))  # 100x100矩阵
            b = torch.randn(100, 100).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            # 执行矩阵乘法 → 触发CUDA核函数
            c = torch.matmul(a, b)  # 结果矩阵 100x100


        except Exception as error:
            l.error(f"预热失败: {str(error)}")
            torch.cuda.empty_cache()  # 显存异常时清空缓存
            continue

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
                f = True
                torch.cuda.empty_cache()  # 释放缓存
                break#跳出
        if f == True:
            continue #此次error，
        # 记录时间戳
        end_time_ns = time.time_ns()
        # 记录持续时间（毫秒）
        end_time = datetime.now().isoformat()

        # 保证采样完整
        time.sleep(2)

        # 计算时间
        duration = round((end_time_ns - start_time_ns)/loop_per_sample, 2)

        other_data = {
            "duration": duration,
            "start_time": start_time,
            "end_time": end_time,
        }

        test_config = op.config

        l.info(f"监测到数据\n{test_config}{other_data}")

        # 解析数据字典
        dictionaries = [test_config, other_data]
        for dictionary in dictionaries:
            for key, value in dictionary.items():
                # 第一次的时候需要初始化
                if key not in records:
                    records[key] = []
                records[key].append(value)

    # 开始写最终的数据
    result_file = join(result_dir, file_name)

    # 写入CSV
    write_csv(result_file, records)


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

    # 解析命令
    if len(sys.argv) < 2:
        logger.warning("没有提供要测试的算子名称和测试次数！将运行默认测试用例!")
    elif len(sys.argv) < 3:
        logger.warning(f"没有指定测试次数！默认每个项目测试{num_samples}次")
    else:
        op_names = sys.argv[1].split(",")
        not_implements = [op_name for op_name in op_names if op_name not in REGISTRY]
        if len(not_implements) > 0:
            logger.error(f"找不到算子或模型{not_implements}，您确定在operator或models模块下实现"
                         f"并在OPERATOR_REGISTRY或MODEL_REGISTRY中注册了它吗？")
            exit(-1)
        try:
            num_samples = int(sys.argv[2])
        except ValueError as e:
            logger.error(f"参数2必须是个整数！而不是{sys.argv[2]}")
            exit(-1)

    logger.info(f"命令行参数解析完成，开始实验，测试的算子或模型有：\n{op_names}")
    logger.info(f"每个算子或模型测试{num_samples}次")

    # 开始主循环
    for op_name in op_names:
        op = REGISTRY[op_name](logger)

        logger.info(f"对{op_name}的实验开始!测试{num_samples}次!")
        logger.info(f"默认设备是{op.device}")
        operation_monitor(
            op,
            op_name,
            logger,
            num_samples,
        )
        logger.info(f"对算子{op_name}的{num_samples}次测试结束!")
        logger.info("--------------------------------------------------------------------------------------------------------------------------------------")#分割

    logger.info("实验结束！")
