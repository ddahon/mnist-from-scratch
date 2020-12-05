import numpy as np
import functions

class NeuralNet:
    def __init__(self, size, learning_rate, layers_activation_function=functions.sigmoid, last_activation_function=functions.softmax):
        
        self.nb_layers = len(size)
        self.learning_rate = learning_rate
        self.layers_activation_function = layers_activation_function
        self.last_activation_function = last_activation_function

        # Weights random initialization
        self.weights = []
        for i in range(self.nb_layers-1):
            size_previous_layer = size[i]
            size_next_layer = size[i+1]
            self.weights.append(np.random.random([size_next_layer, size_previous_layer]))
        
        # Biases random initialization
        self.biases = []
        for layer_size in size[1:]:
            self.biases.append(np.random.random(layer_size))
        print(self.biases)
    # Forward propagate
    # Argument x : training example
    # Returns y_pred : prediction for the input x
    def forward_propagate(self, x):
        a = x
        for current_layer in range(self.nb_layers-1):
            print("Layer number " + str(current_layer))
            z = np.dot(self.weights[current_layer], a) + self.biases[current_layer]
            a = self.layers_activation_function(z)
        y_pred = self.last_activation_function(a) 

        return y_pred
            