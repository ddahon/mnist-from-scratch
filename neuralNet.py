import numpy as np
import functions

class NeuralNet:
    def __init__(self, size, layers_activation_function=functions.sigmoid, last_activation_function=functions.softmax):
        
        self.nb_layers = len(size)-1
        self.layers_activation_function = layers_activation_function
        self.last_activation_function = last_activation_function

        # Weights random initialization
        self.weights = []
        for i in range(self.nb_layers):
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
    # Returns a : activations for each layer
    #         z : pre-activations for each layer
    def forward_propagate(self, x):
        a = [[x]]
        z = [[x]]
        for current_layer in range(self.nb_layers):
            print("weights : " + str(len(self.weights[current_layer])) + "\na : " + str(len(a[-1])))
            res = np.dot(self.weights[current_layer], a[-1])
            z.append(res + self.biases[current_layer])
            if (current_layer == self.nb_layers-1):
                a.append(self.last_activation_function(z[-1]))
            else:
                a.append(self.layers_activation_function(z[-1]))

        return a, z
    
    def back_propagate(self, a, y, z, learning_rate):
        # Last layer update
        delta = np.multiply((a[-1] - y) , z[-1])
        delta_prev = delta

        # Backpropagation
        for current_layer in range(self.nb_layers - 1, 0, -1): 
            delta_prev = delta  
            delta = np.multiply(np.dot(np.transpose(self.weights[current_layer+1]), delta_prev), functions.sigmoid_prime(z[current_layer]))
            dC_db = delta
            dC_dw = np.dot(np.transpose(a[current_layer - 1]), delta)

            # Update of the current layer's weights
            self.weights[current_layer] -= learning_rate * dC_dw

            # Update of the current layer's biases
            self.biases[current_layer] -= learning_rate * dC_db

    def fit(self, epochs, training_examples, training_targets, learning_rate):
        for epoch in range(epochs):
            cost = 0
            for x, y in zip(training_examples, training_targets):
                a, z = self.forward_propagate(x)
                self.back_propagate(a, y, z, learning_rate)
                cost += functions.mse(a[-1], y)

            mean_cost = cost / len(training_examples)
            print("Average cost on epoch " + str(epoch) + " : " + str(mean_cost))
