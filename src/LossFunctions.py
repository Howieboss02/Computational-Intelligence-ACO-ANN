import numpy as np
from Activations import sigmoid

# class CrossEntropy:
#     def categorical_cross_entropy(self, y_true, y_pred):
#         '''
#         y_true - correct label
#         y_pred - label predicted by a model
#         '''
#         return -np.sum(y_true * np.log(y_pred + 10**-100))

class LogLikelihood:
    """
    Class implementing log-likelihood cost function
    """
    def get_cost(data, network):
        """
        :param data:
        :param predictions:
        :return:
        """

        total = 0
        for x, y in data:
            output = network.feedforward(x)
            total += np.dot(y.reshape(y.shape[0]), np.log(output).reshape(output.shape[0]))
        return - (total / len(data))

    def get_delta(a, y, z):
        """
        Method to return delta for the output layer.
        :param a: predicted outcome
        :param y: actual outcome
        :return: delta (error) for output layer
        """
        return a-y

class QuadraticCost:
    """
    Quadratic cost function. ONLY FOR SIGMOID AT OUTPUT LAYER.
    """
    
    def get_cost(data, network):
        """
        The cost calculated over all the data points
        :param data:
        :param network: the neural network
        :return:
        """
        total = 0
        for x, y in data:
            output = network.feedforward(x)
            total += 0.5 * (np.linalg.norm(output - y) ** 2)
        return - (total / len(data))

    def get_delta(a, y, z):
        """
        Method to return delta for the output layer.
        :param a: predicted outcome
        :param y: actual outcome
        :return: delta (error) for output layer
        """
        return (a - y) * sigmoid(z, derivative=True)