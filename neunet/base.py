

class ClassifierModel(object):

    def eval_score(self, x):
        raise NotImplementedError

    def eval_loss(self, batch_x, batch_y):
        raise NotImplementedError

    def list_param_and_grad(self):
        raise NotImplementedError


class ModelTrainer(object):

    def __init__(self, model):
        assert isinstance(model, ClassifierModel)
        self._model = model

    def update_model(self):
        raise NotImplementedError

    def update_data(self, **data):
        raise NotImplementedError


class ModelRender(object):

    def __init__(self, model, trainer):
        assert isinstance(model, ClassifierModel)
        assert isinstance(trainer, ModelTrainer)
        self._model = model
        self._trainer = trainer

    def render_model(self):
        raise NotImplementedError


