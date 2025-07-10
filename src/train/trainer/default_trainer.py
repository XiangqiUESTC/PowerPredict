class DefaultTrainer:
    def __init__(self, data, predictor, config, logger):
        self.data = data
        self.predictor = predictor
        self.config = config
        self.logger = logger


    def build_predictor(self):
        config =  {}
        self.predictor.setup(config)
        pass

    def train(self):
        gpu_info = {}
        self.predictor.setup(gpu_info)
        self.predictor.fit(self.data.input_feature, self.data.output_feature)
