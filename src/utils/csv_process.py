# -*- coding = utf-8 -*-
# @Time :2025/7/7 0:49
import pandas as pd

# # 读取CSV文件
# df = pd.read_csv('../dataset/softmax.csv')
#
# # 假设要拆分的列名为'column_to_split'
# # 这里以示例中的"[129, 134, 129]"为例
# # 首先去除方括号
# df['tensor_shape'] = df['tensor_shape'].str.strip('[]')
#
# # 然后拆分为三列
# split_cols = df['tensor_shape'].str.split(',', expand=True)
# col = ['x', 'y', 'z']
# # 3. 将拆分后的3列插入到前3列
# for i in range(3):
#     df.insert(i,col[i], split_cols[i])
#
# # 删除原始列（如果需要）
# df = df.drop('tensor_shape', axis=1)
#
# # 保存处理后的数据
# df.to_csv('../dataset/softmax_p.csv', index=False)


import pandas as pd

# 读取CSV文件
df = pd.read_csv('../dataset/catMerged.csv')

# 假设'tensor_shape'列包含类似"[[1,2,3],[4,5,6],[7,8,9]]"的字符串
# 1. 去除外层方括号
df['tensor_shapes'] = df['tensor_shapes'].str.strip('[]')

# 2. 先按行拆分（拆分成3行）
rows = df['tensor_shapes'].str.split('],\s*\[', expand=True)

# 3. 对每行再进行拆分（拆分成3列）
all_columns = []
for i in range(3):
    # 去除每行的剩余方括号
    row_data = rows[i].str.strip('[]')
    # 拆分每行的数据
    split_cols = row_data.str.split(',', expand=True)
    # 添加列名
    for j in range(3):
        col_name = f'pos{i+1}{j+1}'  # 例如pos_1_1, pos_1_2等
        df[col_name] = split_cols[j]
        all_columns.append(col_name)

# 删除原始列（如果需要）
df = df.drop('tensor_shapes', axis=1)

# 重新排列列顺序（可选）
df = df[all_columns + [col for col in df.columns if col not in all_columns]]

# 保存处理后的数据
df.to_csv('../dataset/cat_p.csv', index=False)