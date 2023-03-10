import numpy as np
from Activations import sigmoid
from NeuralNetwork import ANN

class LogLikelihood:

    def get_difference(a, y, z):

        return a - y
    
    def get_cost(data, network: ANN):

        total = 0
        for x, y in data:
            output = network.forward_propagate(x)
            total += np.dot(y.reshape(y.shape[0]), np.log(output).reshape(output.shape[0]))
        return - (total / len(data))

    

class QuadraticCost:

    def get_difference(a, y, z):

        return (a - y) * sigmoid(z, derivative=True)
    
    def get_cost(data, network: ANN):
        all = 0
        for x, y in data:
            output = network.forward_propagate(x)
            all += 0.5 * (np.linalg.norm(output - y) ** 2)
        return -(all / len(data))
