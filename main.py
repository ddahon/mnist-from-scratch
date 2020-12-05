from neuralNet import NeuralNet

net = NeuralNet([2, 4, 2], 0.2)

print(net.forward_propagate([3.5, 2.8]))