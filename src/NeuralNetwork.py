# Note: you are free to organize your code in the way you find most convenient.
# However, make sure that when your main notebook is run, it executes the steps indicated in the assignment.
import numpy as np

def train_test_val_split(X, Y, train_size, test_size):
    # compute sizes of sets
    train_size = int(train_size * X.shape[0])
    test_size = int(test_size * X.shape[0])

    # shuffle indexes to get random division into sets
    idx = np.random.permutation(X.shape[0])

    # assign indexes of sets
    train_idx, test_idx, val_idx = idx[:train_size], idx[train_size : train_size + test_size], idx[train_size + test_size:]
    return X[train_idx,:], X[test_idx,:], X[val_idx,:], Y[train_idx,], Y[test_idx,], Y[val_idx,]

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
    def __init__(self, alpha: float = 0, beta: float = 0):
        '''
        alpha - slope of the line for z < 0 for LReLU
        beta - slope for the line for z >= 0 for LReLU
        '''
        self.alpha = alpha
        self.beta = beta



    def softmax(self, z):
        ''' Softmax activation function for output layer
        z - input
        '''
        # subtract max value to get rid of dividing by a large numbers
        e_z = np.exp(z - np.max(z))
        return e_z / e_z.sum(axis=-1, keepdims=True)

    def LReLU(self, z):
        ''' LReLU activation function for hidden layer
        z - input
        '''
        np.where(z < 0, self.alpha * z, self.beta * z)
        return z

class Loss:
    def __init__(self):
        pass
    
    def categorical_cross_entropy(self, y_true, y_pred):
        '''
        y_true - correct label
        y_pred - label predicted by a model
        '''
        return -np.sum(y_true * np.log(y_pred + 10**-100))

class ANN:
    def __init__(self, hidden_layer_sizes: list, lr: float, activations: list, loss_finction: Loss, number_of_features: int = 10):
        '''
        initialize weights using He initialization
        '''
        self.weights = [np.random.normal(loc = 0.0, scale =  2 / (j), size = (i + 1, j)) for i, j in zip([number_of_features] + hidden_layer_sizes[:-1], hidden_layer_sizes)]
        self.lr = lr
        self.activations = activations

    def fit(self, X_train: np.array, y_train: np.array):
        pass

    def predict(self, X: np.array):
        for W_i, activation in zip(self.weights, self.activations):
            # insert a column representing base value
            X = np.insert(X, 0, 1, axis=1)

            X = X @ W_i
            X = activation(X)

        # chose a label with the highest probability
        return X.argmax(axis=1) + 1
    
    def train(self, X):
        #back_propagation()
        pass

    def back_propagation(self):
        pass

class Layer:
    def __init__(self, n: int, function: str):
        self.n = n                  # Number of neurons
        self.function = function    # Activation function



