# -*- coding = utf-8 -*-
# @Time :2025/7/7 0:49
import pandas as pd

# 读取CSV文件
df = pd.read_csv('../dataset/softmax.csv')

# 假设要拆分的列名为'column_to_split'
# 这里以示例中的"[129, 134, 129]"为例
# 首先去除方括号
df['tensor_shape'] = df['tensor_shape'].str.strip('[]')

# 然后拆分为三列
split_cols = df['tensor_shape'].str.split(',', expand=True)
col = ['x', 'y', 'z']
# 3. 将拆分后的3列插入到前3列
for i in range(3):
    df.insert(i,col[i], split_cols[i])

# 删除原始列（如果需要）
df = df.drop('tensor_shape', axis=1)

# 保存处理后的数据
df.to_csv('../dataset/softmax_p.csv', index=False)