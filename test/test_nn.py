

from neunet import BinaryNeuralNetwork, SGDTrainer, MatplotRender

model = BinaryNeuralNetwork(batch_size=3)
trainer = SGDTrainer(model, batch_size=3)
render = MatplotRender(model, trainer).render_model()

