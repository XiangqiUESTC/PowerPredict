import csv
import bisect
import os
from os.path import join, dirname, abspath
from pathlib import Path

from csv_utils import csv_to_dict_list, write_csv


def csv_align(processor_csv_filename, monitor_csv_filename):
    """
        对齐csv数据，其中processor_csv_filename是一个计算过程的数据
        而monitor_csv_filename是监控数据
    """
    print(f"正在处理对齐csv文件{processor_csv_filename},\n监测数据文件为{monitor_csv_filename}")

    with (open(processor_csv_filename, 'r', encoding='utf-8') as file1,
          open(monitor_csv_filename, 'r', encoding='utf-8') as file2):
        processor_csv = csv.reader(file1)
        monitor_data_csv = csv.reader(file2)

        processor_dict_list = csv_to_dict_list(processor_csv)
        monitor_dict_list = csv_to_dict_list(monitor_data_csv)

        # 预处理 monitor 的时间戳列表（已有序）
        monitor_timestamps = monitor_dict_list["timestamp"]

        # 需要对齐的键名
        keys_to_align = {
            "avg_cpu_power_watts": [],
            "max_cpu_power_watts": [],

            "avg_gpu_power_watts": [],
            "max_gpu_power_watts": [],

            "avg_memory_percent": [],
            "max_memory_percent": [],

            "disk_read_bytes": [],
            "disk_write_bytes": [],
        }

        # 遍历 processor 的每一行
        for i in range(len(processor_dict_list["start_time"])):
            # 提取当前行的起始和结束时间
            start_time = processor_dict_list["start_time"][i]
            end_time = processor_dict_list["end_time"][i]

            # 二分查找找到左边界（第一个 >= start_time 的位置）
            left = bisect.bisect_left(monitor_timestamps, start_time)

            # 二分查找找到右边界（第一个 > end_time 的位置）
            right = bisect.bisect_right(monitor_timestamps, end_time)

            # 提取满足条件的 monitor 行索引 [left, right)
            # todo:这个right+1中的+1按理应该去掉，但是有很多空数据，暂时不去掉

            # print(list(range(left, right+1)))
            # 把对应的数据找出来，注意不要越界，并跳过一些空行
            if len(monitor_timestamps) > right >= left > 1:
                for key in monitor_dict_list:
                    rows_for_key = monitor_dict_list[key][left:right + 1]
                    if key == "cpu_power_watts":
                        # 转化为浮点数
                        rows_for_key = [float(row) for row in rows_for_key]

                        # 最大的cpu功率
                        keys_to_align["max_cpu_power_watts"].append(max(rows_for_key))
                        # 平均的cpu功率
                        keys_to_align["avg_cpu_power_watts"].append(round(sum(rows_for_key) / len(rows_for_key), 2))

                    elif key == "gpu_power_watts":
                        # 转化为浮点数
                        rows_for_key = [float(row) for row in rows_for_key]

                        # 最大的gpu功率
                        keys_to_align["max_gpu_power_watts"].append(max(rows_for_key))
                        # 平均的gpu功率
                        keys_to_align["avg_gpu_power_watts"].append(round(sum(rows_for_key) / len(rows_for_key), 2))

                    elif key == "memory_percent":
                        # 转化为浮点数
                        rows_for_key = [float(row) for row in rows_for_key]

                        # 最大的gpu功率
                        keys_to_align["avg_memory_percent"].append(max(rows_for_key))
                        # 平均的gpu功率
                        keys_to_align["max_memory_percent"].append(round(sum(rows_for_key) / len(rows_for_key), 2))

                    elif key == "disk_read_bytes":
                        # 转化为整数
                        rows_for_key = [int(row) for row in rows_for_key]

                        keys_to_align["disk_read_bytes"].append(rows_for_key[-1] - rows_for_key[0])

                    elif key == "disk_write_bytes":
                        # 转化为整数
                        rows_for_key = [int(row) for row in rows_for_key]

                        keys_to_align["disk_write_bytes"].append(rows_for_key[-1] - rows_for_key[0])

                    else:
                        continue
            # 反之，在else中，left >= right
            else:
                for key in keys_to_align:
                    keys_to_align[key].append(None)

        # print(keys_to_align)

        # 终于到了激动人心的写数据环节
        return {
            **processor_dict_list,
            **keys_to_align,
        }


def search_and_align(folder_name, monitor_csv_filename):
    """
        递归地搜索然后找到所有的名为results的文件夹
    """
    results_folders = []

    # 使用 pathlib 处理路径（兼容不同操作系统）
    root_dir = Path(folder_name)

    # 递归遍历所有子目录
    for path in root_dir.rglob("*"):
        if path.is_dir() and path.name == "results":
            results_folders.append(str(path.resolve()))

    # 对每个result_folders下的csv文件进行对齐操作
    for results_folder in results_folders:
        parent_folder = dirname(results_folder)
        # 遍历所有csv文件
        for processor_csv_filename in sorted(os.listdir(results_folder)):
            final_csv = csv_align(join(results_folder, processor_csv_filename), monitor_csv_filename)
            # 写csv
            final_csv_filename = join(parent_folder, "results_aligned", processor_csv_filename)
            print(f"写入文件{final_csv_filename}中!")
            write_csv(final_csv_filename, final_csv)


if __name__ == '__main__':
    """
        对齐数据所用的主函数
    """
    base_path = "data"

    abs_dir_path = join(dirname(abspath(__file__)), base_path)

    folders_to_search = ["算子在CPU上跑-cpu_operator_result_0429", "算子在GPU上跑-gpu_operator_result_0429"]

    monitor_data_csv_filename = join(abs_dir_path, "monitor_result_0429/all_monitor_log.csv")

    for folder in folders_to_search:
        search_and_align(join(abs_dir_path, folder), monitor_data_csv_filename)
