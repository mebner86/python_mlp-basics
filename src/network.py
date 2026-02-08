import random
import numpy as np


class Network(object):
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        """Return the output of the network if ``a`` is input."""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural network outputs the correct number.
        Note that the neural network's output is assumed to be the index of whichever neuron in the
        final layer has the highest activation."""
        test_results = [np.argmax(self.feedforward(x), y) for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        """Returns the vector of partial derivatives \partial C_x / \partial a for the output
        activations"""
        return output_activations - y

def sigmoid(z):
    """The sigmoid function."""
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):
    """ "Derivative of the sigmoid function."""
    return sigmoid(z) * (1 - sigmoid(z))
