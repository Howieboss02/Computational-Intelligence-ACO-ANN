# Note: you are free to organize your code in the way you find most convenient.
# However, make sure that when your main notebook is run, it executes the steps indicated in the assignment.
import numpy as np

def step_function(input: float) -> float:
    return 1 if input >= 0 else 0

class Perceptron:
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

class Activation:
    def __init__(self, K: int = 0, alpha: float = 0, beta: float = 0):
        '''
        K - number of possible classes for softmax
        alpha - slope of the line for z < 0 for LReLU
        beta - slope for the line for z >= 0 for LReLU
        '''
        pass


    def softmax(self, z):
        ''' Softmax activation function for output layer
        z - input
        '''
        pass

    def LReLU(self, z):
        ''' LReLU activation function for hidden layer
        z - input
        '''
        return np.maximum(z, np.zeros(1))

class Loss:
    def __init__(self):
        pass
    
    def categorical_cross_entropy(self, y, K: int):
        '''
        y - input
        K - number of possible classes
        '''
        pass

class ANN:
    def __init__(self, hidden_layer_sizes: list, lr: float, activations: list, loss_finction: str, number_of_features: int = 10):
        '''
        initialize weights using He initialization
        '''
        self.weights = [np.random.normal(loc = 0.0, scale=  2 / (j), size = (i + 1, j)) for i, j in zip([number_of_features] + hidden_layer_sizes[:-1], hidden_layer_sizes)]
        self.lr = lr
        self.activations = activations
        # self.activation_output
        # self.loss_finction

    def fit(self, X_train: np.array, y_train: np.array):
        pass

    def predict(self, X_test: np.array):
        pass

    def feed_foward(self, X):
        X = np.insert(X, 0, 1, axis=1)
        for W_i, activation in zip(self.weights, self.activations):
            print("1: ", X)
            X = np.dot(X, W_i)
            print("2: ", X)
            print("a: ", activation)
            X = activation(X)
            print("3: ", X)

        return X

    def back_propagation(self):
        pass

class Layer:
    def __init__(self, n: int, function: str):
        self.n = n                  # Number of neurons
        self.function = function    # Activation function



