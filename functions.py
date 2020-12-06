import numpy as np 

def relu(x):
    return np.maximum(0, x)

def softmax(x):
    return np.exp(x)/np.sum(np.exp(x))

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def sigmoid_prime(x):
    return sigmoid(x)*(1 - sigmoid(x))

def mse(predictions, targets):
    return np.sum((predictions-targets)**2)/len(predictions)
