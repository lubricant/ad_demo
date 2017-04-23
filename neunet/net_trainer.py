
import numpy as np
import autodiff as ad

from neunet import NeuralNetwork


class SGDTrainer(object):

    def __init__(self, net,
                 batch_size, epoch_num,
                 step=0.01, momentum=0, l1_decay=0, l2_decay=0):

        assert isinstance(net, NeuralNetwork)
        self.net = net

        self.step = step if step else 0.01
        self.momentum = momentum if momentum else 0.
        self.l1_decay = l1_decay if l1_decay else 0.
        self.l2_decay = l2_decay if l2_decay else 0.

        if momentum > 0:
            self.momentum_cache = [np.zeros(w.shape) for w in net.weight]

        assert self.step >= 0 and self.momentum >= 0 and l1_decay >= 0 and l2_decay >= 0

    def update(self, batch):

        net = self.net

        x, y = net.input, net.expect
        weight = net.weight
        loss = net.loss

        batch_num = len(batch)
        layer_num = len(weight)

        loss_sum = 0
        grad_sum = [np.zeros(w.shape) for w in weight]
        for feature, label in batch:
            x.value = feature
            y.value = label

            ad.eval(loss)
            loss_sum += loss.result
            for i in range(layer_num):
                w_grad, = weight[i].gradient
                grad_sum[i] += w_grad

        l1_loss, l2_loss = 0, 0
        l1_decay, l2_decay = self.l1_decay, self.l2_decay
        for i in range(layer_num):
            w = weight[i]
            if l1_decay > 0:
                l1_loss += l1_decay * np.linalg.norm(w.value, 1)
                grad_sum[i] += l1_decay * np.sign(w.value)
            if l2_decay > 0:
                l2_loss += l2_decay * np.linalg.norm(w.value, 2) / 2.
                grad_sum[i] += l2_decay * w.value

        for i in range(layer_num):
            weight_grad = grad_sum[i] / batch_num
            if self.momentum > 0:
                weight_momentum = self.momentum * self.momentum_cache[i] - self.step * weight_grad
                weight[i].value += weight_momentum
                self.momentum_cache[i] = weight_momentum
            else:
                weight[i].value += -self.step * weight_grad

        return (loss_sum + l1_loss + l2_loss) / batch_num
