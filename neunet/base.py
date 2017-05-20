

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


class ModelRender(object):

    def __init__(self, model, trainer):
        assert isinstance(model, ClassifierModel)
        assert isinstance(trainer, ModelTrainer)
        self._model = model
        self._trainer = trainer

    def render_model(self):
        raise NotImplementedError


class DataSet(object):

    def __init__(self, data_list, batch_size, stochastic=True):

        assert data_list is not None and batch_size > 0
        assert data_list.__class__.__len__
        assert data_list.__class__.__getitem__

        batch_num = len(data_list) // batch_size
        self.__batch_size = batch_size

        self.__data_list = data_list
        self.__data_iter = [i for i in range(batch_num)] if batch_num else None

        if stochastic and batch_num:
            from numpy import random as rand
            rand.shuffle(self.__data_iter)

        self.__stochastic = stochastic
        self.__attach_data_list = []

    def attach_data(self, attach_list):
        assert attach_list is not None
        assert attach_list.__class__.__len__
        assert attach_list.__class__.__getitem__

        assert len(self.__data_list) == len(attach_list)
        self.__attach_data_list.append(attach_list)
        return self

    def __iter__(self):

        data_list, attach_data_list = self.__data_list, self.__attach_data_list

        data_iter = self.__data_iter
        data_buff = [None for _ in range(1 + len(attach_data_list))]

        def clip_data(data_index=None):
            if not self.__attach_data_list:
                return data_list if not data_index else data_list[data_index]

            data_buff[0] = data_list[data_index]

            for i in range(len(attach_data_list)):
                attach_list = attach_data_list[i]
                data_buff[i+1] = attach_list if not data_index else attach_list[data_index]

            return tuple(data_buff)

        if data_iter is None:
            yield clip_data()
        else:

            batch_size = self.__batch_size
            batch_stop = batch_size * len(data_iter)

            data_stop = len(data_list)
            data_last = clip_data(slice(batch_stop, data_stop)) if batch_stop < data_stop else None

            stochastic = self.__stochastic
            flip_data = stochastic and not data_stop % 2  # Pseudo-random

            if data_last and flip_data:
                yield data_last

            for i in data_iter:
                batch_offset = i * batch_size
                batch_index = slice(batch_offset, batch_offset + batch_size)
                yield clip_data(batch_index)

            if data_last and not flip_data:
                yield data_last

ds = DataSet([1,2,3,4,5], 2)
it = iter(ds)
try:
    while True:
        val = next(it)
        print(val)
except StopIteration:
    pass