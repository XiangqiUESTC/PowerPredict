import csv
import os
import re
from os.path import dirname

import pandas as pd


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
    num_rows = max([len(v) for v in data.values()], default=0)
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)  # 写入表头

        for i in range(num_rows):
            row = []
            for key in headers:
                # 直接取原始值，不做任何转换
                row.append(data[key][i] if i < len(data[key]) else '')
            writer.writerow(row)


def csv_to_dict_list(csv_reader):
    """
        把一个用csv.reader读取的csv文件转化为字典列表
    """
    headers = next(csv_reader)
    results = {}
    data = [[] for _ in range(len(headers))]
    for row in csv_reader:
        for i, item in enumerate(row):
            data[i].append(item)

    for key, column in zip(headers, data):
        results[key] = column

    return results

def merge_csv_to_pd(file_regex, root_folder):
    """

        Argument
            file_regex: 要读的csv文件
            root_folder: 根目录绝对路径
        Description
            递归地读root_folder下满足正则式file_regex的csv格式文件，并合并为pd对象
        Return
    """
    # 编译正则表达式
    pattern = re.compile(file_regex)
    # 存储所有找到的DataFrame
    all_dataframes = []

    # 递归遍历目录
    for root, _, files in os.walk(root_folder):
        for file in files:
            # 检查文件是否匹配正则表达式
            if pattern.search(file):
                # 构建完整文件路径
                file_path = os.path.join(root, file)
                try:
                    # 读取CSV文件并添加到列表
                    df = pd.read_csv(file_path)
                    all_dataframes.append(df)
                except Exception as e:
                    print(f"Error reading {file_path}: {str(e)}")

    # 合并所有DataFrame
    if all_dataframes:
        return pd.concat(all_dataframes, ignore_index=True)
    else:
        print("No matching CSV files found.")
        return pd.DataFrame()  # 返回空DataFrame