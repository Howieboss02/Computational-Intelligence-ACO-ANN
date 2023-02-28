# Note: you are free to organize your code in the way you find most convenient.
# However, make sure that when your main notebook is run, it executes the steps indicated in the assignment.
import numpy as np

def step_function(input: float) -> float:
    return 1 * (input >= 0)

class Perceptron:
    def __init__(self, lr: float, epochs: int) -> None:
        self.lr = lr
        self.epochs = epochs

    def fit(self, X: np.array, y: np.array) -> list:
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
    def __init__(self):
        pass


    def softmax(self, z, K: int):
        ''' Softmax activation function for output layer
        z - input
        K - number of possible classes
        '''
        pass

    def LReLU(self, z, alpha: float, beta: float):
        ''' LReLU activation function for hidden layer
        z - input
        alpha - slope of the line for z < 0
        beta - slope for the line for z >= 0
        '''
        pass

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
    def __init__(self, hidden_layer_sizes: list, lr: float, activation_hidden: str, activation_output: str, loss_finction: str, number_of_features: int = 10):
        '''
        initialize weights using He initialization
        '''
        self.weights = [np.random.normal(loc = 0.0, scale=  2 / (j - 1), size = (i, j)) for i, j in zip([number_of_features] + hidden_layer_sizes[:-1], hidden_layer_sizes)]
        self.lr = lr
        # self.activation_hidden
        # self.activation_output
        # self.loss_finction

    def fit(self, X_train: np.array, y_train: np.array):
        pass

    def predict(self, X_test: np.array):
        pass

    def feed_foward(self):
        pass

    def back_propagation(self):
        pass

class Layer:
    def __init__(self, n: int, function: str):
        self.n = n                  # Number of neurons
        self.function = function    # Activation function



