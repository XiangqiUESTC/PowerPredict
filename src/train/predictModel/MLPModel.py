import csv
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from tqdm import tqdm
from sklearn.neural_network import MLPRegressor
from sklearn.base import clone
import yaml


class MlpModel:
    def __init__(self,op_name):
        self.config = None
        self.name = op_name
        with open('../config/default.yaml', 'r', encoding='utf-8') as file:
            self.config = yaml.safe_load(file)  # 加载yaml文件
        # 读取数据
        self.data = pd.read_csv(self.config[op_name]['data_path'])
        # 计算目标变量：功耗 (duration * avg_gpu_power * 1e-9)
        self.data['power_consumption'] = self.data['duration'] * self.data['avg_gpu_power'] * 1e-9
        # 选择特征和目标变量
        self.features = self.config[op_name]['input_features']
        self.X = self.data[self.features]
        self.y = self.data['power_consumption']
        self.resultPath = self.config[op_name]['result_path']
        self.csvHead = self.config[op_name]['csv_head']
        # 划分训练集和测试集
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2)
        # 特征标准化
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(self.X_train)
        X_test_scaled = self.scaler.transform(self.X_test)
        # 使用示例
        self.mlp = MLPRegressor(
            hidden_layer_sizes=(10, 5),
            activation='relu',
            solver='adam',
            max_iter=8000,
            verbose=False,
            warm_start=False,
        )
        self.mlp.fit(X_train_scaled, self.y_train)
        # 预测
        y_pred = self.mlp.predict(X_test_scaled)
        # 评估模型
        mse = mean_squared_error(self.y_test, y_pred)
        r2 = r2_score(self.y_test, y_pred)
        print(f"Mean Squared Error: {mse:.4f}")
        print(f"R-squared: {r2:.4f}")
        # 示例使用
        self.plot(self.y_test,y_pred)
    def predict_power_consumption(self,x):#输入为二维特征值数组
        # 标准化
        input_data = pd.DataFrame(x, columns=self.features)
        input_scaled = self.scaler.transform(input_data)
        # 预测
        prediction = self.mlp.predict(input_scaled)
        return prediction

    def plot(self, true_val, pred_val):
        """
        绘制真实值与预测值对比图（带误差分析）
        参数：
            true_val: 真实值数组
            pred_val: 预测值数组
        """
        # 转换为numpy数组确保统一处理
        true_val = np.array(true_val)
        pred_val = np.array(pred_val)
        # 创建索引作为x轴
        plt.figure(figsize=(14, 7))
        # 1. 主图：真实值与预测值曲线
        plt.plot(true_val, 'o-', color='#2c7bb6',
                 label='Actual Power', linewidth=1.5, markersize=6, alpha=0.8)
        plt.plot(pred_val, 's--', color='#d7191c',
                 label='Predicted Power', linewidth=1.5, markersize=5, alpha=0.8)
        # 样式设置
        plt.xlabel('Sample Index', fontsize=12)
        plt.ylabel('Power Consumption', fontsize=12)
        plt.title('Actual vs Predicted Power Consumption\n'+self.name, fontsize=14, pad=20)
        plt.legend(loc='upper right', fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.4)
        # 智能调整y轴范围
        y_min = min(np.min(true_val), np.min(pred_val)) * 0.9
        y_max = max(np.max(true_val), np.max(pred_val)) * 1.1
        plt.ylim(y_min, y_max)
        # 紧凑布局
        plt.tight_layout()
        # 保存图片（可选）
        plt.savefig('../../modelResult/' + self.name+ "_power_comparison.png", dpi=300, bbox_inches='tight')
        plt.show()

def testResult(op_name):
    new_mlpModle = mlpModle(op_name)
    predicted_power = new_mlpModle.predict_power_consumption
    y_true = np.array(new_mlpModle.y_test)
    X_values = np.array(new_mlpModle.X_test)
    with open(new_mlpModle.resultPath, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
                # 写入表头
        writer.writerow(new_mlpModle.csvHead)
                # 逐行写入数据
        prediction = predicted_power(X_values)#二维数组
        for i in range(len(X_values)):
            # 获取当前行的特征、真实值和预测值
            features = X_values[i]  # 形状 (5,)
            true_val = y_true[i]  # 标量
            pred_val = prediction[i]  # 标量
            # 写入一行数据
            writer.writerow([
                *features,  # 解包特征值
                float(true_val),
                float(pred_val)
            ])
if __name__ == '__main__':
    testResult("spmm")
    testResult("softmax")
    testResult("cat")
