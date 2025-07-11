from abc import ABC, abstractmethod


class BasePreprocessor(ABC):
    def __init__(self, config, logger, raw_data):
        self.config = config
        self.logger = logger

        # 在父类初始化方法中根据参数排序
        if config.preprocessor.sort:
            # 创建临时计算列
            raw_data['power_cost'] = raw_data["duration"] * raw_data["avg_gpu_power"] * 1e-9

            # 按临时列排序（默认升序）
            raw_data = raw_data.sort_values(by='power_cost')

        self.raw_data = raw_data

    @abstractmethod
    def process(self):
        pass