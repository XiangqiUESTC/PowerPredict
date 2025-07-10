from train.predictor import PREDICTOR_REGISTRY
from train.preprocessor import PREPROCESSOR_REGISTRY


class DefaultTrainer:
    def __init__(self, raw_data, config, logger):
        # 保存参数
        self.raw_data = raw_data
        self.config = config
        self.logger = logger

        # 初始化数据预处理器
        self.preprocessor = PREPROCESSOR_REGISTRY[config.preprocessor](config)
        # 初始化预测模型
        self.predictor = PREDICTOR_REGISTRY[config.predictor](config)

    def build_predictor(self):
        config =  {}
        self.predictor.setup(config)

    def train(self):
        gpu_info = {}
        self.predictor.setup(gpu_info)
        self.predictor.fit(self.preprocessor.input_feature, self.preprocessor.output_feature)
