"""
    一些预处理函数
"""
import ast
import numpy as np
import pandas as pd


def tensor_shape_split(df, col_name):
    """
    将DataFrame中指定列的数组字符串拆分为多个列

    :param df: 输入的Pandas DataFrame
    :param col_name: 包含数组字符串的列名
    :return: 处理后的新DataFrame
    """
    # 验证输入
    if col_name not in df.columns:
        raise ValueError(f"列 '{col_name}' 不存在于DataFrame中")

    # 解析字符串为数组
    df['_temp_array'] = df[col_name].apply(ast.literal_eval)

    # 检查数组维度是否一致
    shapes = df['_temp_array'].apply(lambda x: np.array(x).shape)
    unique_shapes = shapes.unique()

    if len(unique_shapes) > 1:
        raise ValueError(f"数组形状不一致: 发现 {len(unique_shapes)} 种不同形状")

    # 获取数组形状和维度数
    array_shape = unique_shapes[0]
    ndim = len(array_shape)

    # 创建多级索引用于列名
    indices = np.indices(array_shape).reshape(ndim, -1).T
    col_names = [f"col_{'_'.join(map(str, idx))}" for idx in indices]

    # 将数组展平并拆分为多个列
    df_arrays = pd.DataFrame(
        df['_temp_array'].apply(
            lambda arr: np.array(arr).flatten()
        ).tolist(),
        columns=col_names,
        index=df.index
    )

    # 合并到原始DataFrame
    result = pd.concat([
        df.drop(columns=[col_name, '_temp_array']),
        df_arrays
    ], axis=1)

    return result