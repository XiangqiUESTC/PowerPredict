import time
import datetime
import csv
import subprocess
import threading
from os.path import dirname

import psutil
import os
import sys
from collections import deque


date = datetime.datetime.now().strftime("%Y-%monitors-%d_%H-%M-%S")
# ==== 配置区域 ====
OUTPUT_DIR = f"../results/hardware_{date}"
TOTAL_LOG = os.path.join(OUTPUT_DIR, "total_monitor_log.csv")
CPU_LOG = os.path.join(OUTPUT_DIR, "cpu_util_log.csv")
GPU_LOG = os.path.join(OUTPUT_DIR, "gpu_power_log.csv")
GPU_EXTRA_LOG = os.path.join(OUTPUT_DIR, "gpu_extra_log.csv")
MEM_LOG = os.path.join(OUTPUT_DIR, "memory_usage_log.csv")
DISK_LOG = os.path.join(OUTPUT_DIR, "disk_io_log.csv")

# 采样间隔
fast_interval = 0.01
total_interval = 0.1
save_interval = 5.0

monitoring = True

# ==== 数据缓存 ====
cpu_data = deque()
gpu_data = deque()
gpu_extra_data = deque()
mem_data = deque()
disk_data = deque()
total_data = deque()

# ==== 获取函数 ====
def get_cpu_util():
    try:
        return psutil.cpu_percent(interval=None)
    except Exception:
        return None

def get_gpu_power():
    try:
        output = subprocess.check_output(
            "nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits",
            shell=True, text=True
        )
        power_values = [float(p) for p in output.strip().split('\n') if p.strip()]
        return sum(power_values) / len(power_values) if power_values else None
    except Exception as e:
        print(f"获取GPU功耗出错: {e}")
        return None

def get_gpu_temp_and_util():
    try:
        output = subprocess.check_output(
            "nvidia-smi --query-gpu=temperature.gpu,utilization.gpu --format=csv,noheader,nounits",
            shell=True, text=True
        )
        lines = output.strip().split("\n")
        temps, utils = [], []
        for line in lines:
            temp, util = map(str.strip, line.split(','))
            temps.append(float(temp))
            utils.append(float(util))
        avg_temp = sum(temps) / len(temps) if temps else None
        avg_util = sum(utils) / len(utils) if utils else None
        return avg_temp, avg_util
    except Exception as e:
        print(f"获取GPU温度/利用率出错: {e}")
        return None, None

def get_memory_percent():
    return psutil.virtual_memory().percent

def get_disk_io():
    io = psutil.disk_io_counters()
    return io.read_bytes, io.write_bytes

# ==== 写入函数 ====
def write_csv_rows(filename, headers, data_batch):
    os.makedirs(dirname(filename), exist_ok=True)
    write_header = not os.path.exists(filename) or os.stat(filename).st_size == 0
    with open(filename, 'a', newline='') as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(headers)
        writer.writerows(data_batch)

# ==== 监控线程 ====
def monitor_and_save_cpu(monitor_flag):
    last_save = time.time()
    while monitor_flag["flag"]:
        ts = datetime.datetime.now().isoformat()
        val = get_cpu_util()
        cpu_data.append([ts, val])
        if time.time() - last_save >= save_interval:
            write_csv_rows(CPU_LOG, ["timestamp", "cpu_util_percent"], list(cpu_data))
            cpu_data.clear()
            last_save = time.time()
        time.sleep(fast_interval)

def monitor_and_save_gpu(monitor_flag):
    last_save = time.time()
    while monitor_flag["flag"]:
        ts = datetime.datetime.now().isoformat()
        val = get_gpu_power()
        gpu_data.append([ts, val])
        if time.time() - last_save >= save_interval:
            write_csv_rows(GPU_LOG, ["timestamp", "gpu_power_watts"], list(gpu_data))
            gpu_data.clear()
            last_save = time.time()
        time.sleep(fast_interval)

def monitor_and_save_gpu_extra(monitor_flag):
    last_save = time.time()
    while monitor_flag["flag"]:
        ts = datetime.datetime.now().isoformat()
        temp, util = get_gpu_temp_and_util()
        gpu_extra_data.append([ts, temp, util])
        if time.time() - last_save >= save_interval:
            write_csv_rows(GPU_EXTRA_LOG, ["timestamp", "gpu_temp_C", "gpu_util_percent"], list(gpu_extra_data))
            gpu_extra_data.clear()
            last_save = time.time()
        time.sleep(fast_interval)

def monitor_and_save_mem(monitor_flag):
    last_save = time.time()
    while monitor_flag["flag"]:
        ts = datetime.datetime.now().isoformat()
        val = get_memory_percent()
        mem_data.append([ts, val])
        if time.time() - last_save >= save_interval:
            write_csv_rows(MEM_LOG, ["timestamp", "mem_percent"], list(mem_data))
            mem_data.clear()
            last_save = time.time()
        time.sleep(fast_interval)

def monitor_and_save_disk(monitor_flag):
    last_save = time.time()
    while monitor_flag["flag"]:
        ts = datetime.datetime.now().isoformat()
        read, write = get_disk_io()
        disk_data.append([ts, read, write])
        if time.time() - last_save >= save_interval:
            write_csv_rows(DISK_LOG, ["timestamp", "disk_read_bytes", "disk_write_bytes"], list(disk_data))
            disk_data.clear()
            last_save = time.time()
        time.sleep(fast_interval)

def monitor_and_save_total(monitor_flag):
    last_save = time.time()
    while monitor_flag["flag"]:
        ts = datetime.datetime.now().isoformat()
        mem = get_memory_percent()
        read, write = get_disk_io()
        gpu = get_gpu_power()
        cpu = get_cpu_util()
        total_data.append([ts, mem, read, write, gpu, cpu])
        if time.time() - last_save >= save_interval:
            write_csv_rows(TOTAL_LOG,
                           ["timestamp", "mem_percent", "disk_read_bytes", "disk_write_bytes", "gpu_power_watts", "cpu_util_percent"],
                           list(total_data))
            total_data.clear()
            last_save = time.time()
        time.sleep(total_interval)

# ==== 启动 ====
def monitor_main(logger, flag):
    monitor_flag = {"flag": True}
    logger.info("正在启动第三方数据采集线程...")
    threads = [
        threading.Thread(target=monitor_and_save_cpu, args=(monitor_flag,)),
        threading.Thread(target=monitor_and_save_gpu, args=(monitor_flag,)),
        threading.Thread(target=monitor_and_save_gpu_extra, args=(monitor_flag,)),
        threading.Thread(target=monitor_and_save_mem, args=(monitor_flag,)),
        threading.Thread(target=monitor_and_save_disk, args=(monitor_flag,)),
        threading.Thread(target=monitor_and_save_total, args=(monitor_flag,)),
    ]
    for t in threads:
        t.start()

    logger.info("第三方设备数据采集进程启动！")

    # 如果flag为True，一直保持运行，轮询式运行，否则停止所有进程
    while flag["flag"]:
        time.sleep(1)

    logger.info("收到中断信号，正在停止采集...")
    # 设置终止信号，终止各个线程
    monitor_flag["flag"] = False

    # 等待进程结束
    for t in threads:
        t.join()

    # 处理最后的数据
    if cpu_data:
        write_csv_rows(CPU_LOG, ["timestamp", "cpu_util_percent"], list(cpu_data))
    if gpu_data:
        write_csv_rows(GPU_LOG, ["timestamp", "gpu_power_watts"], list(gpu_data))
    if gpu_extra_data:
        write_csv_rows(GPU_EXTRA_LOG, ["timestamp", "gpu_temp_C", "gpu_util_percent"], list(gpu_extra_data))
    if mem_data:
        write_csv_rows(MEM_LOG, ["timestamp", "mem_percent"], list(mem_data))
    if disk_data:
        write_csv_rows(DISK_LOG, ["timestamp", "disk_read_bytes", "disk_write_bytes"], list(disk_data))
    if total_data:
        write_csv_rows(TOTAL_LOG,
                       ["timestamp", "mem_percent", "disk_read_bytes", "disk_write_bytes", "gpu_power_watts", "cpu_util_percent"],
                       list(total_data))

    logger.info("所有数据保存完毕。程序退出。")
    sys.exit(0)
