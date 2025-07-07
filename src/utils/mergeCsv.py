# -*- coding = utf-8 -*-
# @Time :2025/7/7 22:32
import pandas as pd

# 读取两个CSV文件
df1 = pd.read_csv('../dataset/cat.csv')
df2 = pd.read_csv('../dataset/cat2.csv')
df3 = pd.read_csv('../dataset/cat3.csv')

# 垂直合并（堆叠行）
merged_df = pd.concat([df1, df2,df3], ignore_index=True)

# 保存合并后的文件
merged_df.to_csv('../dataset/catMerged.csv', index=False)