
import numpy as np
import autodiff as ad

from neunet import *


class SGDTrainer(ModelTrainer):

    def __init__(self, network_model, batch_size, epoch_num=-1,
                 step=0.01, momentum=0, l1_decay=0, l2_decay=0):

        assert batch_size > 0 and epoch_num is not None

        super().__init__(network_model)
        self._data = None
        self._iter = None
        self._epoch = epoch_num
        self._batch = batch_size

        self._step = step if step else 0.01
        self._momentum = momentum if momentum else 0.
        self._l1_decay = l1_decay if l1_decay else 0.
        self._l2_decay = l2_decay if l2_decay else 0.

        if momentum > 0:
            self._momentum_cache = {}

        assert self._step >= 0 and self._momentum >= 0 and l1_decay >= 0 and l2_decay >= 0

    def update_data(self, feature_set, label_set):
        if self._iter:
            del self._iter

        self._data = DataSet(feature_set, self._batch).attach_data(label_set)
        self._iter = iter(self._data)

    def update_model(self):

        if not self._data:
            return None

        def next_batch():
            try:
                assert self._iter is not None
                return next(self._iter)

            except StopIteration:

                if self._epoch > 0:
                    self._epoch -= 1
                    if not self._epoch:
                        raise StopIteration

                self._iter = iter(self._data)

        network = self._model
        data, label = next_batch()
        assert len(data) == len(label)

        batch_size = self._batch
        step, momentum = self._step, self._momentum
        l1_decay, l2_decay = self._l1_decay, self._l2_decay

        l1_loss, l2_loss = 0, 0
        data_loss = network.eval_loss(data, label) / batch_size

        for param, grad in network.list_param_and_grad():
            param_val = param.value
            param_grad = grad / batch_size

            l1_grad, l2_grad = 0, 0
            if l1_decay > 0:
                l1_loss += l1_decay * np.linalg.norm(param_val, 1)
                l1_grad = l1_decay * np.sign(param_val)
            if l2_decay > 0:
                l2_loss += l2_decay * np.linalg.norm(param_val, 2) / 2.
                l2_grad = l2_decay * param_val

            param_grad += l1_grad
            param_grad += l2_grad

            if momentum > 0:
                momentum_cache = self._momentum_cache
                param_momentum = momentum * momentum_cache[param] - step * param_grad
                momentum_cache[param] = param_momentum
                param.value += param_momentum
            else:
                param.value += -step * param_grad

        return data_loss, l1_loss, l2_loss

