

class DataSet(object):

    def __init__(self, data_list, batch_size, stochastic=True):

        assert data_list is not None and batch_size > 0
        assert data_list.__class__.__len__
        assert data_list.__class__.__getitem__ is not None

        batch_num = len(data_list) // batch_size
        self.__batch_size = batch_size

        self.__data_list = data_list
        self.__data_iter = [i for i in range(batch_num)] if batch_num else None
        if stochastic and batch_num:
            from numpy import random as rand
            rand.shuffle(self.__data_iter)

        self.__stochastic = stochastic

    def __iter__(self):

        data_list, data_iter = self.__data_list, self.__data_iter

        if data_iter is None:
            yield data_list
        else:

            batch_size = self.__batch_size
            batch_stop = batch_size * len(data_iter)

            data_stop = len(data_list)
            data_last = data_list[batch_stop: data_stop] if batch_stop < data_stop else None

            stochastic = self.__stochastic
            flip_data = stochastic and not data_stop % 2  # Pseudo-random

            if data_last and flip_data:
                yield data_last

            for i in data_iter:
                batch_offset = i * batch_size
                yield data_list[batch_offset: batch_offset + batch_size]

            if data_last and not flip_data:
                yield data_last


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
        self.__model = model

    def update_model(self, batch_x, batch_y):
        raise NotImplementedError


class ModelRender(object):

    def __init__(self, model, trainer):
        assert isinstance(model, ClassifierModel)
        assert isinstance(trainer, ModelTrainer)
        self.__model = model
        self.__trainer = trainer

    def render_model(self):
        pass

