import pandas as pd

from train.core.base_preprocessor import BasePreprocessor
from utils.preprocess_util import tensor_shape_split


class SoftmaxPreprocessor(BasePreprocessor):
    def __init__(self, config, logger, raw_data):
        super(SoftmaxPreprocessor, self).__init__(config, logger, raw_data)
        self.input_feature = None
        self.output_feature = None

    def process(self):
        raw_data = self.raw_data
        # 调用处理形状的工具函数
        shapes = tensor_shape_split(raw_data, "tensor_shape")

        other_features = raw_data[["avg_temperature", "avg_gpu_memory_used"]]

        shapes_reset = shapes.reset_index(drop=True)
        other_features_reset = other_features.reset_index(drop=True)

        self.input_feature = pd.concat([shapes_reset, other_features_reset],axis=1)

        self.output_feature = raw_data["duration"] * raw_data["avg_gpu_power"] * 1e-9