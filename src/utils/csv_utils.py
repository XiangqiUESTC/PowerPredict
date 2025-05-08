import csv
import os
from os.path import dirname


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