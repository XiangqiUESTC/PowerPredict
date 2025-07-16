import os
from datetime import datetime
from os.path import join
from sklearn.preprocessing import StandardScaler

import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
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
            hidden_layer_sizes=(128,128,128,128,128,128),
            activation='relu',
            solver='adam',
            max_iter=40000,
            verbose=False,
            warm_start=False,
        )

    def fit(self):
        input_feature = self.preprocessor.input_feature
        output_feature = self.preprocessor.output_feature

        input_feature = pd.DataFrame(input_feature)
        output_feature = pd.Series(output_feature) if not isinstance(output_feature, pd.Series) else output_feature

        # 第二步：归一化特征X（整体归一化）
        scaler = StandardScaler()
        X_normalized = pd.DataFrame(
            scaler.fit_transform(input_feature),
            columns=input_feature.columns,
            index=input_feature.index
        )

        # 第三步：分割数据集（使用归一化后的X和原始y）
        X_train, X_test, y_train, y_test = train_test_split(
            X_normalized,
            output_feature,
            test_size=0.2,
            # random_state=42  # 建议取消注释以保持可复现性
        )

        # 第四步：保留原始分割索引
        test_indices = X_test.index

        # 第五步：获取对应的原始X数据（未归一化）
        X_test_raw = input_feature.loc[test_indices]

        # 增大1000倍，训练更准确
        y_train = y_train * 1000
        y_test = y_test * 1000
        print("训练集大小:", X_train.shape, y_train.shape)
        print("测试集大小:", X_test.shape, y_test.shape)

        self.model.fit(X_train, y_train)

        # 预测
        y_pred = self.model.predict(X_test)
        y_test = y_test / 1000
        y_pred = y_pred / 1000

        # 评估模型
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # 确保使用相同的索引
        output_feature = self.preprocessor.output_feature  # 保留原始索引

        # 计算绝对百分比误差（保留原始索引）
        abs_errors_percent = pd.Series(
            np.abs((y_test - y_pred) / y_test),
            index=y_test.index,
            name="absolute percent error"
        )

        # 计算MAPE（两种方法结果应一致）
        mape = abs_errors_percent.mean() * 100
        mape2 = mean_absolute_percentage_error(y_test, y_pred)

        # 创建预测值Series（强制使用相同索引）
        # y_pred_series = pd.Series(
        #     y_pred,
        #     index=y_test.index,  # 关键点：对齐索引
        #     name='predict cost'
        # )

        # 合并数据（确保所有组件有相同索引）
        # output_csv = pd.concat([
        #     X_test,
        #     y_test.rename("real cost"),
        #     # y_pred_series,
        #     (abs_errors_percent * 100).round(2).astype(str) + '%'
        # ], axis=1)
        output_csv = pd.concat([
            X_test_raw.reset_index(drop=True),  # 丢弃原始索引
            y_test.rename("real cost").reset_index(drop=True),
            pd.Series(y_pred, name="predict cost").reset_index(drop=True),
            pd.Series(
                (abs_errors_percent * 100).round(2).astype(str) + '%',
                name="error %"
            ).reset_index(drop=True)
        ], axis=1)

        # 保存训练数据的csv
        result_folder = join(self.config.project_abs_path, self.config.result_folder)
        t = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        os.makedirs(result_folder, exist_ok=True)
        output_csv.to_csv(
            join(result_folder, f"{self.config.task_name}-{t}.csv"),
            index=False)

        steps = list(range(0, len(y_test)))
        plt.plot(steps, y_test.to_numpy(), 'o-', color='#2c7bb6',
                 label='Actual Power', linewidth=1.5, markersize=6, alpha=0.8)

        plt.plot(y_pred, 's--', color='#d7191c',
                 label='Predicted Power', linewidth=1.5, markersize=5, alpha=0.8)
        # 样式设置
        plt.xlabel('Sample Index', fontsize=12)
        plt.ylabel('Power Consumption(Watt)', fontsize=12)
        plt.title(f'{self.config.task_name} Actual vs Predicted Power Consumption\n' , fontsize=14, pad=8)
        plt.legend(loc='upper right', fontsize=10)
        plt.show()

        self.logger.info(f"均方误差（Mean Squared Error）:: {mse:.4f}")
        self.logger.info(f"R-squared: {r2:.4f}")
        self.logger.info(f"平均绝对误差（Mean Absolute Error）: {mae:.4f}")

        self.logger.info(f"平均百分比绝对误差（Mean Absolute Percentage Error）: {mape}%")

