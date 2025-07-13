from os.path import join

import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt
import pandas as pd

class MLP:
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.model = None
        self.info = None
        self.preprocessor = None

    def setup(self, preprocessor, info):
        """
            info是gpu相关信息（或者其他信息），用于构建模型
        """
        self.preprocessor = preprocessor
        self.info = info
        self.model = MLPRegressor(
            hidden_layer_sizes=(64, 64),
            activation='relu',
            solver='adam',
            max_iter=10000,
            verbose=False,
            warm_start=False,
        )

    def fit(self):
        input_feature = self.preprocessor.input_feature
        output_feature = self.preprocessor.output_feature

        self.model.fit(input_feature, output_feature)

        # 预测
        y_pred = self.model.predict(input_feature)
        # 评估模型
        mse = mean_squared_error(output_feature, y_pred)
        mae = mean_absolute_error(output_feature, y_pred)
        r2 = r2_score(output_feature, y_pred)

        # 确保使用相同的索引
        output_feature = self.preprocessor.output_feature  # 保留原始索引

        # 计算绝对百分比误差（保留原始索引）
        abs_errors_percent = pd.Series(
            np.abs((output_feature - y_pred) / output_feature),
            index=output_feature.index,
            name="absolute percent error"
        )

        # 计算MAPE（两种方法结果应一致）
        mape = abs_errors_percent.mean() * 100
        mape2 = mean_absolute_percentage_error(output_feature, y_pred)

        # 创建预测值Series（强制使用相同索引）
        y_pred_series = pd.Series(
            y_pred,
            index=output_feature.index,  # 关键点：对齐索引
            name='predict cost'
        )

        # 合并数据（确保所有组件有相同索引）
        output_csv = pd.concat([
            self.preprocessor.input_feature,
            output_feature.rename("real cost"),
            y_pred_series,
            (abs_errors_percent * 100).round(2).astype(str) + '%'
        ], axis=1)

        output_csv.to_csv("output_results.csv", index=False)

        steps = list(range(0, len(output_feature)))
        plt.plot(steps, output_feature, 'o-', color='#2c7bb6',
                 label='Actual Power', linewidth=1.5, markersize=6, alpha=0.8)

        plt.plot(y_pred, 's--', color='#d7191c',
                 label='Predicted Power', linewidth=1.5, markersize=5, alpha=0.8)
        # 样式设置
        plt.xlabel('Sample Index', fontsize=12)
        plt.ylabel('Power Consumption', fontsize=12)
        plt.title(f'{self.config.task_name} Actual vs Predicted Power Consumption\n' , fontsize=14, pad=8)
        plt.legend(loc='upper right', fontsize=10)
        plt.show()

        self.logger.info(f"均方误差（Mean Squared Error）:: {mse:.4f}")
        self.logger.info(f"R-squared: {r2:.4f}")
        self.logger.info(f"平均绝对误差（Mean Absolute Error）: {mae:.4f}")

        self.logger.info(f"平均百分比绝对误差（Mean Absolute Percentage Error）: {mape}%")

