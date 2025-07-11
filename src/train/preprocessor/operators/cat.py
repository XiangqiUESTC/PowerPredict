import pandas as pd


class CatPreprocessor:
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.input_feature = None
        self.output_feature = None

    def process(self, raw_data):
        # 假设'tensor_shape'列包含类似"[[1,2,3],[4,5,6],[7,8,9]]"的字符串
        # 1. 去除外层方括号
        raw_data['tensor_shapes'] = raw_data['tensor_shapes'].str.strip('[]')

        # 2. 先按行拆分（拆分成3行）
        rows = raw_data['tensor_shapes'].str.split('],\s*\[', expand=True)

        # 3. 对每行再进行拆分（拆分成3列）
        all_columns = []
        for i in range(3):
            # 去除每行的剩余方括号
            row_data = rows[i].str.strip('[]')
            # 拆分每行的数据
            split_cols = row_data.str.split(',', expand=True)
            # 添加列名
            for j in range(3):
                col_name = f'pos{i + 1}{j + 1}'  # 例如pos_1_1, pos_1_2等
                raw_data[col_name] = split_cols[j]
                all_columns.append(col_name)

        # 删除原始列（如果需要）
        raw_data = raw_data.drop('tensor_shapes', axis=1)

        # 重新排列列顺序（可选）
        raw_data = raw_data[all_columns + [col for col in raw_data.columns if col not in all_columns]]

        self.input_feature = raw_data[["pos11", "pos12", "pos13", "pos21", "pos22", "pos23", "pos31", "pos32", "pos33", "dim"]]
        self.output_feature = raw_data["duration"]*raw_data["avg_gpu_power"]*1e-9

        print(self.input_feature)
        print(self.output_feature)