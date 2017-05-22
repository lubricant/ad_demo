

from neunet import BinaryNeuralNetwork, SGDTrainer, MatplotRender

model = BinaryNeuralNetwork(batch_size=12)
trainer = SGDTrainer(model, batch_size=12, momentum=0.1, l2_decay=0.001)
render = MatplotRender(model, trainer, interval=1).render_model()





