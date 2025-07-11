from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt

class MLP:
    def __init__(self, config,logger):
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
            hidden_layer_sizes=(128, 256, 512),
            activation='relu',
            solver='adam',
            max_iter=7000,
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
        r2 = r2_score(output_feature, y_pred)
        print(f"Mean Squared Error: {mse:.4f}")
        print(f"R-squared: {r2:.4f}")
        plt.plot(output_feature, 'o-', color='#2c7bb6',
                 label='Actual Power', linewidth=1.5, markersize=6, alpha=0.8)
        plt.plot(y_pred, 's--', color='#d7191c',
                 label='Predicted Power', linewidth=1.5, markersize=5, alpha=0.8)
        # 样式设置
        plt.xlabel('Sample Index', fontsize=12)
        plt.ylabel('Power Consumption', fontsize=12)
        plt.title('Actual vs Predicted Power Consumption\n' , fontsize=14, pad=20)
        plt.legend(loc='upper right', fontsize=10)
        plt.show()

