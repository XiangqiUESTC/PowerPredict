from train.predictor import PREDICTOR_REGISTRY
from train.preprocessor import PREPROCESSOR_REGISTRY


class DefaultTrainer:
    def __init__(self, raw_data, config, logger):
        # 保存参数
        self.raw_data = raw_data
        self.config = config
        self.logger = logger

        # 初始化数据预处理器
        self.preprocessor = PREPROCESSOR_REGISTRY[config.preprocessor](config, logger)
        # 初始化预测模型
        self.predictor = PREDICTOR_REGISTRY[config.predictor](config, logger)

    def build_predictor(self):
        """
            考虑到预测模型的构建可能需要GPU相关数据，所以这里可能要收集GPU相关数据信息，然后对预测模型进行完全的构建
            所以在预测模型初始化的时候其实时没有完全初始化的，还要调用predictor的setup方法

            同时把preprocessor也交给predictor方便其进行数据获取
        """
        info =  {}
        self.predictor.setup(self.preprocessor, info)

    def train(self):
        """
            :return:
        """
        gpu_info = {}

        # 装配predictor
        self.predictor.setup(self.preprocessor, gpu_info)

        #
        self.preprocessor.process(self.raw_data)

        self.predictor.fit()
