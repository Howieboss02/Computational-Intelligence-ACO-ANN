# This file contains the Perceptron class used for the XOR problem.
import numpy as np
from Activations import step_function

class Perceptron:
    '''
    A single perceptron to be used in a neural network. This is used for the XOR problem.
    :param lr: learning rate
    :param epochs: number of epochs to train for
    '''

    def __init__(self, lr: float, epochs: int) -> None:
        self.lr = lr
        self.epochs = epochs

    def train(self, X: np.array, y: np.array) -> list:
        weights = np.zeros((X.shape[1] + 1, 1))
        error_list = []

        for epoch in range(self.epochs):

            error = 0

            for xi, yi in zip(X, y):
                xi = np.insert(xi, 0, 1).reshape((-1, 1))
                z = np.dot(xi.T, weights)
                y_hat = step_function(z)
                loss = yi - y_hat
                if (loss != 0): error += 1
                weights += xi * self.lr * loss

            error_list.append(error)

        self.weights = weights
        return error_list
