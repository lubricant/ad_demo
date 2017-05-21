

from neunet import BinaryNeuralNetwork, SGDTrainer, MatplotRender

# model = BinaryNeuralNetwork(batch_size=12)
# trainer = SGDTrainer(model, batch_size=12, l2_decay=0.01)
# render = MatplotRender(model, trainer).render_model()


if __name__ == '__main__':
    import numpy as np
    x = np.array([[2., 3.]])
    y = np.array([1])
    m = BinaryNeuralNetwork(1, rand_seed=2)
    l = m.eval_loss(x, y)

    pg = m.list_param_and_grad()[0]
    print(pg)

    h = 10.e-5
    # check x[0]
    x0_l, x0_h = x.copy(), x.copy()
    x0_l[0] -= h
    x0_h[0] += h
    l_l, l_h = m.eval_loss(x0_l, y), m.eval_loss(x0_h, y)
    print((l_h-l_l)/(2*h))



