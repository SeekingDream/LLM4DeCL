


class AbstractMethod:
    def __init__(self, train_data, val_data, test_data, config):
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.config = config
