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


# 读取数据
data = pd.read_csv('../../dataset/spmm.csv')

# 计算目标变量：功耗 (duration * avg_gpu_power * 1e-9)
data['power_consumption'] = data['duration'] * data['avg_gpu_power'] * 1e-9

# 选择特征和目标变量
features = ['m', 'n', 'p', 'nnz', 'sparsity']
X = data[features]
y = data['power_consumption']

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

# 特征标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# 使用示例
mlp = MLPRegressor(
    hidden_layer_sizes=(10, 5),
    activation='relu',
    solver='adam',
    max_iter=8000,
    verbose=False,
    warm_start=False,
)
mlp.fit(X_train_scaled, y_train)
# 预测
y_pred = mlp.predict(X_test_scaled)
print(y_pred)
print(y_test)

# 评估模型
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse:.4f}")
print(f"R-squared: {r2:.4f}")


def predict_power_consumption(m, n, p, nnz, sparsity):
    # 创建输入数据框
    input_data = pd.DataFrame([[m, n, p, nnz, sparsity]], columns=features)

    # 标准化
    input_scaled = scaler.transform(input_data)

    # 预测
    prediction = mlp.predict(input_scaled)

    return prediction[0]
# 示例使用
m, n, p, nnz, sparsity = 6016, 6016, 5888, 17711104,0.5
predicted_power = predict_power_consumption(m, n, p, nnz, sparsity)
y_predicted =  []
X_values = np.array(X)
predicted_set = []
with open('spmm_predictions.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    # 写入表头
    writer.writerow(['m', 'n', 'p', 'nnz', 'sparsity', 'real_power','predicted_power'])
    # 逐行写入数据
    for x_i, y_real in zip(X_values, y):
        prediction = predict_power_consumption(*x_i)
        predicted_set.append(prediction)
        writer.writerow(list(x_i) + [y_real,prediction])

plt.plot(y, label='Actual Power', color='#2c7bb6', linewidth=2, marker='o', markersize=5)
plt.plot(predicted_set, label='Predicted Power', color='#d7191c', linestyle='--', linewidth=2, marker='x', markersize=5)

# 添加标注和样式
plt.xlabel('Sample Index', fontsize=12)
plt.ylabel('Power Consumption', fontsize=12)
plt.title('Actual vs Predicted Power Consumption', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.show()
