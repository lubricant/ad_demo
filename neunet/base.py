
class BinaryClassifierModel(object):

    def predict(self, x, y):
        return 1 if x == y else 0


class ModelTrainer(object):

    def __init__(self, model):

        assert isinstance(model, BinaryClassifierModel)
        self.model = model

    def update(self, batch):
        pass


class ModelRender(object):

    def __init__(self, model, trainer):
        assert isinstance(model, BinaryClassifierModel)
        assert isinstance(trainer, ModelTrainer)
        self.model = model
        self.trainer = trainer

    def render(self):
        pass
