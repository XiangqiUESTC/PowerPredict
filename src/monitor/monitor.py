"""
    包括主要的监测逻辑函数和监控线程
"""
from operators import *
from models import *
from utils.csv_utils import write_csv
from utils.device import is_device_avail_on_torch


from datetime import datetime
from os.path import join, dirname, abspath
import torch
import threading
import time

# 同时注册算子和模型


def run_and_monitor(args, logger, op_names, num_samples):
    """
        ARGS:
            args: operation是基本的算子或者模型,同时是base_operation的实现类的实例对象
            logger: 算子的名称,用于命名最后的文件
            op_names: 日志实例对象
            num_samples: 该算子要测试的默认数据组数
        DESCRIPTION:
            创建，分别监控该算子在运算时的CPU和GPU数据，分析并合并输出为csv
        """
    # 记录开始时间
    # 注册所有的算子和模型
    REGISTRY = {
        **OPERATOR_REGISTRY,
        **MODEL_REGISTRY,
    }

    # 开始主循环
    for op_name in op_names:
        operator = REGISTRY[op_name](args, logger)
        logger.info(f"对{op_name}的实验开始!测试{num_samples}次!")
        logger.info(f"执行操作的设备是{operator.device}")

        file_name = op_name + ".csv"

        # 文件日期
        file_date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        # csv中间文件夹
        temp_dir = join(abspath(dirname(dirname(abspath(__file__)))), "temp")

        result_dir = join(abspath(dirname(dirname(abspath(__file__)))), "results", result_folder)

        # 最终结果
        records = {}

        # 获取当前的设备设置信息
        device = operator.device
        device_avail = is_device_avail_on_torch(device)

        # 如果设备正常，运行测试
        if device_avail:
            # 循环num_sample次
            for j in range(num_samples):
                try:
                    # 开始测试
                    logger.info(f"Test Case {j + 1} for {op_name}: Starting monitoring and computation...")
                    # 生成此次测试的配置
                    _ = operator.generate_config()
                    # 装配数据
                    try:
                        operator.setup()
                    except Exception as error:
                        if isinstance(error, torch.OutOfMemoryError):
                            torch.cuda.empty_cache()
                            logger.info("显存溢出，清除显存")
                        else:
                            logger.exception(error)
                        continue

                    # 预热GPU
                    try:
                        for i_ in range(preheat):
                            # 创建两个随机矩阵（尺寸可根据需要调整）
                            a = torch.randn(100, 100).to(
                                torch.device("cuda" if torch.cuda.is_available() else "cpu"))  # 100x100矩阵
                            b = torch.randn(100, 100).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
                            # 执行矩阵乘法 → 触发CUDA核函数
                            c = torch.matmul(a, b)  # 结果矩阵 100x100

                    except Exception as error:
                        logger.error(f"预热失败: {str(error)}")
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
                            operator.execute()
                        except Exception as error:
                            logger.error(error)
                            logger.exception(error)
                            f = True
                            torch.cuda.empty_cache()  # 释放缓存
                            # 跳出
                            break
                    if f:
                        # 此次error
                        continue
                    torch.cuda.empty_cache()  # 释放缓存
                    # 记录时间戳
                    end_time_ns = time.time_ns()
                    # 记录持续时间（毫秒）
                    end_time = datetime.now().isoformat()

                    # 结束监控线程
                    stop_gpu["stop"] = True
                    gpu_thread.join()

                    # 保证采样完整
                    time.sleep(4)

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
                        "gpu_model": get_gpu_model(op.device),
                    }

                    # 计算时间
                    duration = round((end_time_ns - start_time_ns) / loop_per_sample, 2)

                    other_data = {
                        "duration": duration,
                        "start_time": start_time,
                        "end_time": end_time,
                    }

                    test_config = op.config

                    logger.info(f"监测到数据\n{test_config}{other_data}{gpu_data}")

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
            logger.error(f"{device}不可用,已跳过测试")
        logger.info(f"对算子{op_name}的{num_samples}次测试结束!")
        logger.info(
            "--------------------------------------------------------------------------------------------------------------------------------------")  # 分割

    logger.info("实验结束！")

def operation_monitor(a, b, c, d):
    pass

def cpu_monitor_thread():
    pass

def gpu_monitor_thread():
    pass

def disk_monitor_thread():
    pass

def memory_monitor_thread():
    pass