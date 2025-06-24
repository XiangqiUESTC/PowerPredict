import time
import datetime
import csv
import subprocess
import threading
import psutil
import os
import sys
from collections import deque

# ==== 配置区域 ====
OUTPUT_DIR = "./"
TOTAL_LOG = os.path.join(OUTPUT_DIR, "total_monitor_log.csv")
CPU_LOG = os.path.join(OUTPUT_DIR, "cpu_util_log.csv")
GPU_LOG = os.path.join(OUTPUT_DIR, "gpu_power_log.csv")
MEM_LOG = os.path.join(OUTPUT_DIR, "memory_usage_log.csv")
DISK_LOG = os.path.join(OUTPUT_DIR, "disk_io_log.csv")

# 采样间隔
fast_interval = 0.01       # 设备采样频率（秒）
total_interval = 0.1      # 总表采样频率（秒）
save_interval = 5.0      # 每个设备写入频率（秒）

monitoring = True

# ==== 数据缓存 ====
cpu_data = deque()
gpu_data = deque()
mem_data = deque()
disk_data = deque()
total_data = deque()

# ==== 设备采集函数 ====
def get_cpu_util():
    try:
        return psutil.cpu_percent(interval=None)
    except Exception:
        return None

# def get_gpu_power():
#     try:
#         output = subprocess.check_output(
#             "npu-smi -m | awk '{print $5\"C\",$9\"W\",$18\"MiB\",$19\"MiB\",$20\"%\"}'",
#             shell=True, text=True
#         )
#         return output.split("\n")[1].split()[1]
#     except Exception:
#         return None

def get_memory_percent():
    return psutil.virtual_memory().percent

def get_disk_io():
    io = psutil.disk_io_counters()
    return io.read_bytes, io.write_bytes

# ==== 数据写入函数 ====
def write_csv_rows(filename, headers, data_batch):
    write_header = not os.path.exists(filename) or os.stat(filename).st_size == 0
    with open(filename, 'a', newline='') as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(headers)
        writer.writerows(data_batch)

# ==== 各设备线程 ====
def monitor_and_save_cpu():
    last_save = time.time()
    while monitoring:
        ts = datetime.datetime.now().isoformat()
        val = get_cpu_util()
        cpu_data.append([ts, val])
        if time.time() - last_save >= save_interval:
            batch = list(cpu_data)
            cpu_data.clear()
            write_csv_rows(CPU_LOG, ["timestamp", "cpu_util_percent"], batch)
            last_save = time.time()
        time.sleep(fast_interval)


def get_gpu_power():
    try:
        # 执行 npu-smi info 命令获取原始输出
        output = subprocess.check_output("npu-smi info", shell=True, text=True)

        # 解析输出，提取所有NPU的功耗值
        power_values = []
        temperature_values = []
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
                    temperature = parts[2].split()[1]  # 提取功耗值(如"93.9")
                    temperature_values.append(float(temperature))
                    # print("gpu功耗: ", power)

        # 计算平均功耗(如果有多个NPU)
        if power_values:
            return sum(power_values) / len(power_values),sum(temperature_values) / len(temperature_values)
        return None

    except Exception as e:
        print(f"获取NPU功耗出错: {e}")
        return None


def monitor_and_save_gpu():
    last_save = time.time()
    while monitoring:
        ts = datetime.datetime.now().isoformat()
        gpu_power,gpu_temp = get_gpu_power()
        gpu_data.append([ts, gpu_power,gpu_temp])
        if time.time() - last_save >= save_interval:
            batch = list(gpu_data)
            gpu_data.clear()
            write_csv_rows(GPU_LOG, ["timestamp", "gpu_power_watts"], batch)
            last_save = time.time()
        time.sleep(fast_interval)




def monitor_and_save_mem():
    last_save = time.time()
    while monitoring:
        ts = datetime.datetime.now().isoformat()
        val = get_memory_percent()
        mem_data.append([ts, val])
        if time.time() - last_save >= save_interval:
            batch = list(mem_data)
            mem_data.clear()
            write_csv_rows(MEM_LOG, ["timestamp", "mem_percent"], batch)
            last_save = time.time()
        time.sleep(fast_interval)

def monitor_and_save_disk():
    last_save = time.time()
    while monitoring:
        ts = datetime.datetime.now().isoformat()
        read, write = get_disk_io()
        disk_data.append([ts, read, write])
        if time.time() - last_save >= save_interval:
            batch = list(disk_data)
            disk_data.clear()
            write_csv_rows(DISK_LOG, ["timestamp", "disk_read_bytes", "disk_write_bytes"], batch)
            last_save = time.time()
        time.sleep(fast_interval)

def monitor_and_save_total():
    last_save = time.time()
    while monitoring:
        ts = datetime.datetime.now().isoformat()
        mem = get_memory_percent()
        read, write = get_disk_io()
        gpu_power,gpu_temp = get_gpu_power()
        cpu = get_cpu_util()
        total_data.append([ts, mem, read, write, gpu_power, gpu_temp, cpu])
        if time.time() - last_save >= save_interval:
            batch = list(total_data)
            total_data.clear()
            write_csv_rows(TOTAL_LOG, ["timestamp", "mem_percent", "disk_read_bytes", "disk_write_bytes", "gpu_power_watts","gpu_temperature", "cpu_util_percent"], batch)
            last_save = time.time()
        time.sleep(total_interval)

# ==== 启动程序 ====
if __name__ == "__main__":
    try:
        print(" 正在启动设备采集线程...")
        threads = [
            threading.Thread(target=monitor_and_save_cpu),
            threading.Thread(target=monitor_and_save_gpu),
            threading.Thread(target=monitor_and_save_mem),
            threading.Thread(target=monitor_and_save_disk),
            threading.Thread(target=monitor_and_save_total),
        ]

        for t in threads:
            t.start()

        print("📡 正在采集设备数据，按 Ctrl+C 停止...")
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        print("\n 收到中断信号，正在停止采集...")
        monitoring = False
        for t in threads:
            t.join()

        # 最后一批未写入的数据（防丢失）
        if cpu_data:
            write_csv_rows(CPU_LOG, ["timestamp", "cpu_util_percent"], list(cpu_data))
        if gpu_data:
            write_csv_rows(GPU_LOG, ["timestamp", "gpu_power_watts","gpu_temperature"], list(gpu_data))
        if mem_data:
            write_csv_rows(MEM_LOG, ["timestamp", "mem_percent"], list(mem_data))
        if disk_data:
            write_csv_rows(DISK_LOG, ["timestamp", "disk_read_bytes", "disk_write_bytes"], list(disk_data))
        if total_data:
            write_csv_rows(TOTAL_LOG, ["timestamp", "mem_percent", "disk_read_bytes", "disk_write_bytes", "gpu_power_watts", "cpu_util_percent"], list(total_data))

        print(" 所有数据保存完毕。程序退出。")
        sys.exit(0)
