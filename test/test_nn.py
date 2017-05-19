
from test.nn.nn_render import ModelRender
from test.nn.nn_model import NeuralNetwork

# model = NeuralNetwork(momentum=0.1, l2_decay=0.001)
# render = ModelRender(model)

from neunet import NeuralNetwork

net = NeuralNetwork(3)
print(net)


