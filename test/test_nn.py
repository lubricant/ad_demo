

from neunet import BinaryNeuralNetwork, SGDTrainer, MatplotRender

model = BinaryNeuralNetwork(batch_size=12)
trainer = SGDTrainer(model, batch_size=12, l2_decay=0.01)
render = MatplotRender(model, trainer, interval=0.5).render_model()





