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

class ANN:
    def __init__(self):
        pass