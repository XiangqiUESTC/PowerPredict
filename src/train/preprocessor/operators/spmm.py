from train.core.base_preprocessor import BasePreprocessor


class SpmmPreprocessor(BasePreprocessor):
    def __init__(self, config, logger, raw_data):
        super(SpmmPreprocessor, self).__init__(config, logger, raw_data)
        self.input_feature = None
        self.output_feature = None

    def process(self):
        raw_data = self.raw_data

        self.input_feature = raw_data[["m", "n", "p", "sparsity", "avg_temperature", "avg_gpu_memory_used"]]
        self.output_feature = raw_data["duration"]*raw_data["avg_gpu_power"]*1e-9