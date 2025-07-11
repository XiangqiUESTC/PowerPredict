from abc import ABC, abstractmethod


class BasePreprocessor(ABC):
    def __init__(self, config, logger, raw_data):
        self.config = config
        self.logger = logger
        self.raw_data = raw_data
        pass

    @abstractmethod
    def process(self):
        pass