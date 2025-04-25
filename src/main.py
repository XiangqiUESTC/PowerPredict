import sys
from os.path import dirname, join, abspath
import time
import threading
import subprocess
import csv
from datetime import datetime
from operators import *
from models import *
import os
from utils.logger import Logger


# ----------------- GPU 监控线程 -----------------
def gpu_monitor_thread_func(logfile, stop_flag, l):
    """
    ARGS:
        logfile: gpu监测信息的输出文件的绝对路径
        stop_flag: stop_flag是一个字典，字典属于非基本变量，通过字典里面的值来在外部控制进程的结束
        l: 日志实例对象
    DESCRIPTION:
        gpu监控线程函数
    """
    # 新增目录创建逻辑
    os.makedirs(dirname(logfile), exist_ok=True)
    # 如果是昆仑芯NPU，则取消注释以下代码，并注释掉原有的监测GPU的代码
    # l.info("MONITOR-NPU-THREAD")
    # with open(logfile, 'w') as f:
    #     kml_proc = subprocess.Popen(["kml-smi", "--query", "power", "--interval", "1000"], stdout=kml_log)
    #     while not stop_flag["stop"]:
    #         line = kml_proc.stdout.readline()
    #         # l.info("LINE: ", line)
    #         if line:
    #             f.write(line)
    #             f.flush()
    #     proc.terminate()
    # GPU监测代码
    with open(logfile, 'w') as f:
        f.write("timestamp  power.draw [W] util [%] memory [MiB]\n")
        while not stop_flag["stop"]:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            try:
                output = subprocess.check_output([
                    'nvidia-smi',
                    '--query-gpu=power.draw,utilization.gpu,memory.used',
                    '--format=csv,noheader,nounits'
                ])
                line = output.decode('utf-8').strip()
                f.write(f"{timestamp},{line}\n")
                f.flush()
            except Exception as e:
                l.error("GPU monitoring error:", e)
                break


# ----------------- CPU 监控线程（使用 turbostat） -----------------
def cpu_monitor_thread_func(logfile, stop_flag, l):
    """
    ARGS:
        logfile: cpu监测信息的输出文件的绝对路径
        stop_flag: stop_flag是一个字典，字典属于非基本变量，通过字典里面的值来在外部控制进程的结束
        l: 日志实例对象
    DESCRIPTION:
        cpu监控线程函数
    """
    # 新增目录创建逻辑
    os.makedirs(dirname(logfile), exist_ok=True)
    # 'turbostat' '--quiet' '--Summary' '--interval' '1'
    cmd = ['turbostat', '--quiet', '--Summary', '--interval', '1']
    with open(logfile, 'w') as f:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, universal_newlines=True, bufsize=1)
        while not stop_flag["stop"]:
            line = proc.stdout.readline()
            if line:
                # l.info("LINE: ", line)
                f.write(line)
                f.flush()
        proc.terminate()


def write_csv(filename, data):
    """
    ARGS:
        filename: 要写入的文件名
        data: 数据,一个字典,键名为列名,键值为列值的列表
    DESCRIPTION:
        读写csv的函数
    """
    # 新增目录创建逻辑
    os.makedirs(dirname(filename), exist_ok=True)
    # 获取所有键（列名）
    headers = data.keys()
    # 获取行数（以最长的列表为准）
    num_rows = max(len(v) for v in data.values())
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)  # 写入表头

        for i in range(num_rows):
            row = []
            for key in headers:
                # 直接取原始值，不做任何转换
                row.append(data[key][i] if i < len(data[key]) else '')
            writer.writerow(row)


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
        operation.setup()

        # 预热GPU
        time.sleep(2)
        for _ in range(preheat):
            operation.execute()

        # 启动GPU监控线程
        gpu_log = join(temp_dir, f'gpu_{operation_name}_{file_date}.csv')
        stop_gpu = {"stop": False}
        gpu_thread = threading.Thread(target=gpu_monitor_thread_func, args=(gpu_log, stop_gpu, l))
        gpu_thread.start()

        # 启动CPU监控线程
        # cpu_log = join(temp_dir, f'cpu_{operation_name}_{file_date}.csv')
        # stop_cpu = {"stop": False}
        # cpu_thread = threading.Thread(target=cpu_monitor_thread_func, args=(cpu_log, stop_cpu, l))
        # cpu_thread.start()

        nanoseconds = time.time_ns()
        start_time = nanoseconds // 1_000_000
        # 重复执行，不断采样
        for _ in range(loop_per_sample):
            operation.execute()
        nanoseconds = time.time_ns()
        end_time = nanoseconds // 1_000_000

        # Todo 此处执行完之后为什么不立刻停止线程？不怕采集到非算子运行时的数据吗？
        # 保证采样完整
        time.sleep(2)
        # 停止监测线程

        stop_gpu["stop"] = True
        gpu_thread.join()
        # stop_cpu["stop"] = True
        # cpu_thread.join()

        # 分析 GPU 数据
        powers, utils, mems = [], [], []
        with open(gpu_log, 'r') as f:
            # 跳过文件头
            next(f)
            for data_line in f:
                data_item = data_line.strip().split(',')
                if len(data_item) == 4:
                    powers.append(float(data_item[1]))
                    utils.append(float(data_item[2]))
                    mems.append(float(data_item[3]))
        # 取平均,保留两位小数
        avg_power = round(sum(powers) / len(powers), 2) if powers else 0
        avg_util = round(sum(utils) / len(utils), 2) if utils else 0
        avg_mem = round(sum(mems) / len(mems), 2) if mems else 0
        # 有时候没有读到任何一条数据,避免max空列表出错,加入数据0
        powers.append(0)
        utils.append(0)
        mems.append(0)
        max_power = round(max(powers), 2)
        max_util = round(max(utils), 2)
        max_mem = round(max(mems), 2)

        gpu_data = {
            'avg_gpu_power': avg_power,
            'arg_gpu_util': avg_util,
            'arg_gpu_mem': avg_mem,
            'max_gpu_power': max_power,
            'max_gpu_util': max_util,
            'max_gpu_mem': max_mem,
        }
        # 计算时间
        duration = round((end_time - start_time)/loop_per_sample, 2)

        other_data = {
            "duration": duration
        }

        test_config = op.config

        l.info(f"监测到数据\n{test_config}{gpu_data}{other_data}")

        # 分析 CPU 数据（取 PKG(W) 字段平均）
        # cpu_powers = []
        # with open(cpu_log, 'r') as f:
        #     for line in f:
        #         # l.info("line: ", line)
        #         if 'Avg_MHz' not in line:
        #             parts = line.strip().split()
        #             try:
        #                 cpu_powers.append(float(parts[22]))
        #             except:
        #                 pass
        # avg_cpu = round(sum(cpu_powers) / len(cpu_powers), 2) if cpu_powers else 0
        # cpu_data = {
        #     "avg_cpu": avg_cpu
        # }

        # 解析数据字典
        dictionaries = [test_config, gpu_data, other_data]
        for dictionary in dictionaries:
            for key, value in dictionary.items():
                # 第一次的时候需要初始化
                if key not in records:
                    records[key] = []
                else:
                    records[key].append(value)

    # 开始写最终的数据
    result_file = join(result_dir, file_name)

    # 写入CSV
    write_csv(result_file, records)


# ----------------- 主函数 -----------------
if __name__ == '__main__':
    # 参数设置
    # op_names = [
    #     "avg_pooling", "conv", "elu", "linear_layer", "max_pooling", "relu", "silu", "leaky_relu",
    #     "spmm", "flatten", "cat", "lay_norm", "embedding", "positional_encoding", "roi_align",
    #     "nms", "add", "softmax", "lstm"
    # ]
    # 同时注册算子和模型
    REGISTRY = {
        **OPERATOR_REGISTRY,
        **MODEL_REGISTRY,
    }

    # 默认测试所有
    op_names = [
        *REGISTRY.keys()
    ]

    num_samples = 5

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
        op = REGISTRY[op_name]()

        logger.info(f"对{op_name}的实验开始!测试{num_samples}次!")
        operation_monitor(
            op,
            op_name,
            logger,
            num_samples,
        )
        logger.info(f"对算子{op_name}的{num_samples}次测试结束!")
        logger.info("------------------------------------------------------------------------")#分割

    logger.info("实验结束！")
