
class BinaryClassifierModel(object):

    def predict(self, x):
        pass


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
