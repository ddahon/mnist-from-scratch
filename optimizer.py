import functions
from neuralNet import *

class Optimizer:
    
    def __init__(self, training_examples, training_targets, neural_net, learning_rate=0.1):
        self.training_examples = training_examples
        self.training_targets = training_targets
        self.neural_net = neural_net
        self.learning_rate = learning_rate

    # Stochastic gradient descent
    def back_propagate(self, a, y, z):
        # Last layer update
        delta = np.multiply((a[-1] - y) , z[-1])
        delta_prev = delta

        # Backpropagation
        for current_layer in range(self.neural_net.nb_layers - 1, 0, -1): 
            delta_prev = delta  
            delta = np.multiply(np.dot(np.transpose(self.neural_net.weights[current_layer+1]), delta_prev), functions.sigmoid_prime(z[current_layer]))
            dC_db = delta
            dC_dw = np.dot(np.transpose(a[current_layer - 1]), delta)

            # Update of the current layer's weights
            self.neural_net.weights[current_layer] -= self.learning_rate * dC_dw

            # Update of the current layer's biases
            self.neural_net.biases[current_layer] -= self.learning_rate * dC_db

    def fit(self, epochs):
        for epoch in range(epochs):
            cost = 0
            for x, y in zip(self.training_examples, self.training_targets):
                a, z = self.neural_net.forward_propagate(x)
                self.back_propagate(a, y, z)
                cost += functions.mse(a[-1], y)

            mean_cost = cost / len(self.training_examples)
            print("Average cost on epoch " + str(epoch) + " : " + str(mean_cost))
