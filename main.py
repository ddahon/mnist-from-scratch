from optimizer import Optimizer
from neuralNet import NeuralNet
import functions

size = [1, 2, 3]
training_examples = [1, 4, 5]
training_targets = [2, 8, 10]
model = NeuralNet(size, last_activation_function=functions.relu)
model.fit(10, training_examples, training_targets, 0.1)
