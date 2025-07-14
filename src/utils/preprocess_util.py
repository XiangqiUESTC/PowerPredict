"""
    一些预处理函数
"""
import pandas as pd
import numpy as np
import ast
import itertools


def tensor_shape_split(df, col_name):
    """
    将DataFrame中特定列的数组字符串，根据其形状拆分成多个新列。

    :param df: pandas DataFrame，包含源数据。
    :param col_name: 字符串，df中需要被拆分的列名。该列的每一行都应是同规格数组的字符串表示。
    :return: 一个新的pandas DataFrame，包含了从原数组拆分出的所有新列。
    """
    # 步骤 1: 使用 ast.literal_eval 将字符串安全地转换为 Python 的列表对象
    # 我们不能使用 eval()，因为它不安全。ast.literal_eval 只能处理 Python 的字面量。
    temp_col = df[col_name].apply(ast.literal_eval)

    # 步骤 2: 获取数组的形状
    # 因为所有行的数组规格都相同，所以我们只需要检查第一行即可
    if temp_col.empty:
        return pd.DataFrame() # 如果输入为空，则返回一个空的DataFrame
    first_item_list = temp_col.iloc[0]
    shape = np.array(first_item_list).shape

    # 步骤 3: 根据形状生成新列的名称
    # 例如，对于 shape (3, 3)，我们需要生成索引 (0,0), (0,1), (0,2), (1,0)...
    # itertools.product 可以完美地实现这个功能
    index_tuples = itertools.product(*[range(dim) for dim in shape])

    # 将索引元组格式化为 "col_i_j_k..." 的形式
    new_col_names = [f"col_{'_'.join(map(str, idx))}" for idx in index_tuples]

    # 步骤 4: 扁平化数组并创建新的DataFrame
    # 对 temp_col 中的每个列表，我们将其转换为 numpy 数组并调用 flatten() 方法
    # 这会返回一个 Series，其中每个元素都是一个一维数组
    flattened_data = temp_col.apply(lambda x: np.array(x).flatten())

    # 将这个包含一维数组的Series转换为一个新的DataFrame
    # .tolist() 将Series中的所有数组转换为一个列表的列表
    # 然后pandas的构造函数会很高效地创建出新的DataFrame
    new_df = pd.DataFrame(
        flattened_data.tolist(),
        columns=new_col_names,
        index=df.index  # 保持与原DataFrame相同的索引
    )

    return new_df