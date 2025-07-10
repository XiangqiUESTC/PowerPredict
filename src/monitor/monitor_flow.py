"""
    包括主要的监测逻辑函数和监控线程
"""
from time import sleep

import psutil

from monitor.operators import *
from monitor.models import *
from monitor.monitors.gpu import get_gpu_model, get_gpu_info_by_pynvml
from monitor.monitors.cpu import get_cpu_info
from monitor.monitors.disk import get_disk_info
from monitor.monitors.memory import get_memory_info
from utils.device import is_device_avail_on_torch
from utils.device import is_device_gpu
from utils.csv_utils import write_csv


from datetime import datetime
from os.path import join
import torch
import threading
import time


def run_all_and_monitor(args, logger, op_names, num_samples, result_folder):
    """
        Description
            在run_all_and_monitor主程序中，会运行各个根据args装配各个算子的配置，并将每个算子运行num_samples次
            算子的配置包含其基本的各类测试模式生成测试配置所需的参数、所需要的收集的数据的集合（GPU、CPU、Disk、Memory等）
            还有所测试的设备名等

            对于所需要的收集的数据，每类数据都会创建一个线程用于收集数据
        Arguments
            args: args是命令行参数
            logger: logger是日志器
            op_names: op_names是所有的要测试的算子名称
            num_samples: num_samples是所有算子要测试的次数
            result_folder: result_folder是结果文件夹
        Returns
            None
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

        # 最终结果
        records = {}

        # 获取运行算子所必要的信息：所指定的设备、一次实验重复的次数、实验间的间隔时间、预热次数
        # 设备信息：
        device = operator.device
        device_avail = is_device_avail_on_torch(device)
        # 重复次数：
        loop_per_sample = operator.loop_per_sample
        sample_time = operator.sample_time
        # 实验间的间隔时间：
        sleep_time = operator.sleep_time
        # 预热次数：
        preheat = operator.preheat

        # 如果设备正常，运行测试
        if device_avail:
            # 循环num_sample次
            for j in range(num_samples):
                try:
                    # 开始测试
                    logger.info(f"开始对{op_name}的第{j + 1}/{num_samples}次测试...")

                    # 生成此次测试的配置
                    _ = operator.generate_config()

                    # 装配数据，此处要进行错误捕获，特别注意显存溢出错误，由于不知道模式，所以不会因为一次显存爆了就终止之后的实验，仍应继续实验
                    try:
                        operator.setup()
                    except Exception as error:
                        logger.error("根据装配测试所需的数据时出错！原因如下：")
                        logger.exception(error)
                        if isinstance(error, torch.OutOfMemoryError):
                            logger.info("将清除显存")
                            torch.cuda.empty_cache()
                        continue

                    # 预热GPU
                    try:
                        for _ in range(preheat):
                            # 创建两个随机矩阵（尺寸可根据需要调整）
                            a = torch.randn(100, 100).to(
                                torch.device("cuda" if torch.cuda.is_available() else "cpu"))  # 100x100矩阵
                            b = torch.randn(100, 100).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
                            # 执行矩阵乘法 → 触发CUDA核函数
                            # 结果矩阵 100x100
                            c = torch.matmul(a, b)
                    except Exception as error:
                        logger.error(f"预热失败，原因是：")
                        logger.exception(error)
                        # 显存异常时清空缓存
                        torch.cuda.empty_cache()
                        continue

                    # 准备正式开始实验，准备好监控线程
                    threads = []
                    flags = []
                    datas = []
                    params = []
                    # 判断是否启动cpu监控
                    if operator.cpu_info:
                        cpu_data = {}
                        cpu_flag = {"flag": True}
                        cpu_params = {
                            "process": psutil.Process(),
                            "interval": operator.cpu_monitor_interval
                        }
                        cpu_thread = threading.Thread(target=cpu_monitor_thread,args=(cpu_data, cpu_flag, cpu_params, logger))

                        # 在threads、flags、datas和params添加内容
                        threads.append(cpu_thread)
                        flags.append(cpu_flag)
                        datas.append(cpu_data)
                        params.append(cpu_params)
                    # 判断是否启动内存监控
                    if operator.memory_info:
                        memory_data = {}
                        memory_flag = {"flag": True}
                        memory_params = {
                            "process": psutil.Process(),
                            "interval": operator.memory_monitor_interval
                        }
                        memory_thread = threading.Thread(target=memory_monitor_thread, args=(memory_data, memory_flag, memory_params, logger))

                        # 在threads、flags、datas和params添加内容
                        threads.append(memory_thread)
                        flags.append(memory_flag)
                        datas.append(memory_data)
                        params.append(memory_params)
                    # 判断是否启动gpu监控
                    if operator.gpu_info:
                        gpu_data = {}
                        gpu_flag = {"flag": True}
                        gpu_params = {
                            "gpu": operator.gpu,
                            "device": operator.device,
                            "interval": operator.gpu_monitor_interval,
                        }
                        gpu_thread = threading.Thread(target=gpu_monitor_thread, args=(gpu_data, gpu_flag, gpu_params, logger))

                        # 在threads、flags、datas和params添加内容
                        threads.append(gpu_thread)
                        flags.append(gpu_flag)
                        datas.append(gpu_data)
                        params.append(gpu_params)
                    # 判断是否启动硬盘数据监控
                    if operator.disk_info:
                        disk_data = {}
                        disk_flag = {"flag": True}
                        disk_params = {
                            "interval": operator.disk_monitor_interval,
                        }
                        disk_thread = threading.Thread(target=disk_monitor_thread,
                                                      args=(disk_data, disk_flag, disk_params, logger))

                        # 在threads、flags、datas和params添加内容
                        threads.append(disk_thread)
                        flags.append(disk_flag)
                        datas.append(disk_data)
                        params.append(disk_params)

                    # 启动各个监控线程
                    for thread in threads:
                        thread.start()

                    # 记录时间戳
                    start_time = datetime.now().isoformat()
                    # 记录持续时间（毫秒）
                    start_time_ns = time.time_ns()
                    # 重复执行，不断采样
                    f = False
                    # 统计执行次数
                    execute_time = 0
                    while True:
                        try:
                            operator.execute()
                        except Exception as error:
                            logger.error(error)
                            logger.exception(error)
                            f = True
                            torch.cuda.empty_cache()  # 释放缓存
                            # 跳出
                            break
                        execute_time += 1
                        current_time_ns = time.time_ns()
                        # 如果设置了采样时间，以采样时间为准
                        if sample_time:
                            passed_sample_time = (current_time_ns - start_time_ns) / 1e6
                            if passed_sample_time > sample_time:
                                break
                        # 否则以重复次数为准
                        elif loop_per_sample:
                            if execute_time > loop_per_sample:
                                break
                        else:
                            logger.error(
                                "至少应该指定一次测试的测试总时间或者次数！而现在sample_time和loop_per_sample均为None！")
                            exit(-1)
                    if f:
                        # 此次error
                        continue
                    torch.cuda.empty_cache()  # 释放缓存
                    # 记录时间戳
                    end_time_ns = time.time_ns()
                    # 记录持续时间（毫秒）
                    end_time = datetime.now().isoformat()

                    # 结束各个进程
                    for flag in flags:
                        flag["flag"] = False
                    # 同步
                    for thread in threads:
                        thread.join()

                    # 保证采样完整
                    time.sleep(sleep_time)

                    # 计算时间
                    duration = round((end_time_ns - start_time_ns) / execute_time, 2)

                    other_data = {
                        "duration": duration,
                        "start_time": start_time,
                        "end_time": end_time,
                    }

                    test_config = operator.config

                    logger.info(f"所有监测子进程正常退出，监测到数据\n{test_config}{other_data}{datas}")

                    # 解析数据字典
                    dictionaries = [test_config, other_data, *datas]
                    for dictionary in dictionaries:
                        for k, v in dictionary.items():
                            # 第一次的时候需要初始化
                            if k not in records:
                                records[k] = []
                            records[k].append(v)

                except Exception as error:
                    logger.error(f"{op_name}第{j + 1}/{loop_per_sample}次重复测试失败，原因是：\n")
                    logger.exception(error)
            # 算子运行结束，开始写最终的数据
            result_file = join(result_folder, file_name)
            # 写入CSV
            write_csv(result_file, records)
        else:
            logger.error(f"所指定的{device}不可用,已跳过测试")
        logger.info(f"对算子{op_name}的{num_samples}次测试结束!\n\n")

    logger.info("实验结束！")

"""---------------------------------------------------以下是监控线程---------------------------------------------------"""
def cpu_monitor_thread(data, flag, params, logger):
    """
        Description
            用于监控cpu线程
        Arguments
            data :
            flag :
            params:
            logger:
        Returns
            None
    """
    logger.info("CPU监控线程启动！收集数据中...")
    infos = []
    while flag["flag"]:
        cpu_info = get_cpu_info()
        infos.append(cpu_info)
        sleep(params["interval"])
    logger.debug(f"CPU监控线程共收集到{len(infos)}条数据")
    cpu_percents = [info["cpu_percent"] for info in infos]
    data.update(
        {
            "max_cpu_percent": round(max(cpu_percents), 2),
            "avg_cpu_percent": round(sum(cpu_percents)/len(cpu_percents), 2),
         }
    )
    logger.info(f"CPU监控线程正常退出，处理了收集到的{len(infos)}条数据...")

def memory_monitor_thread(data, flag, params, logger):
    """
        Description

        Arguments
            data :
            flag :
            params:
            logger:
        Returns
            None
        """
    logger.info("内存监控线程启动！收集数据中...")
    main_process = params["process"]
    infos = []
    while flag["flag"]:
        infos.append(get_memory_info(main_process))
        sleep(params["interval"])
    memories = [info["memory"] for info in infos]
    data.update({
        "max_memory": round(max(memories), 2),
        "avg_memory": round(sum(memories)/len(memories), 2),
    })
    logger.debug(f"内存监控线程共监控到{len(infos)}条数据")
    logger.info(f"内存监控线程正常退出，处理了收集到的{len(infos)}条数据...")

def gpu_monitor_thread(data, flag, params, logger):
    """
    Description

    Arguments
        data :
        flag :
        params:
        logger:
    Returns
        None
    """
    gpu = params["gpu"]
    device = params["device"]

    if not is_device_gpu(gpu):
        logger.error(f"{gpu}不是一个有效的gpu设备！gpu监控线程退出！")
        exit(-1)

    if is_device_gpu(device):
        if device != gpu:
            logger.warning(f"当device是gpu类型时，配置项device应该和配置项gpu一致，现在device是{device}而gpu是{gpu}")

    logger.info("GPU监控线程启动！收集数据中...")

    infos = []
    while flag["flag"]:
        # 获取gpu信息
        info = get_gpu_info_by_pynvml(gpu)
        sleep(params["interval"])
        if info is not None:
            infos.append(info)
    logger.debug(f"GPU监控线程收集到{len(infos)}条数据")
    # 收到终止信号，开始处理数据
    if len(infos) != 0:
        # 解析GPU功耗数据
        powers = [info["gpu_power"] for info in infos]
        utils = [info["gpu_utilization"] for info in infos]
        memory_used = [info["gpu_memory_used"] for info in infos]
        temperature = [info["gpu_temperature"] for info in infos]

        # 取平均,保留两位小数
        avg_power = round(sum(powers) / len(powers), 2) if powers else 0
        max_power = round(max(powers, default=0), 2)

        # 取平均,保留两位小数
        avg_utils = round(sum(utils) / len(utils), 2) if utils else 0
        max_utils = round(max(utils, default=0), 2)

        # 取平均,保留两位小数
        avg_memory_used = round(sum(memory_used) / len(memory_used), 2) if memory_used else 0
        max_memory_used = round(max(memory_used, default=0), 2)

        # 温度取平均,保留两位小数
        avg_temperature = round(sum(temperature) / len(temperature), 2) if temperature else 0
        max_temperature = round(max(temperature, default=0), 2)

        data.update({
            "max_gpu_power": max_power,
            "avg_gpu_power": avg_power,
            "max_gpu_util": max_utils,
            "avg_gpu_utils": avg_utils,
            "max_gpu_memory_used": max_memory_used,
            "avg_gpu_memory_used": avg_memory_used,
            "avg_temperature": avg_temperature,
            "max_temperature": max_temperature,
            "gpu_model": get_gpu_model(gpu)
        })
        logger.info(f"GPU监控线程正常退出，处理了收集到的{len(infos)}条数据...")

def disk_monitor_thread(data, flag, params, logger):
    """
        Description

        Arguments
            data :
            flag :
            params:
            logger:
        Returns
            None
        """
    logger.info("硬盘监控线程启动！收集数据中...")
    infos = []
    while flag["flag"]:
        infos.append(get_disk_info())
        sleep(params["interval"])
    logger.debug(f"硬盘监控线程收集到{len(infos)}条数据")
    read_bytes = [info["read_bytes"] for info in infos]
    write_bytes = [info["write_bytes"] for info in infos]
    total_read_bytes = round((read_bytes[-1] - read_bytes[0])/(1024**2))
    total_write_bytes = round((write_bytes[-1] - write_bytes[0])/(1024**2))
    data.update({
        "read_bytes": total_read_bytes,
        "write_bytes": total_write_bytes,
    })
    logger.info(f"硬盘监控线程正常退出，处理了收集到的{len(infos)}条数据...")