import numpy as np

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
        # e_z = np.exp(z - np.max(z))
        # return e_z / e_z.sum(axis=-1, keepdims=True)
        # r = z.copy()
        # r = r.squeeze()
        r = z
        exp_x = np.exp(r - np.max(r))  # subtract the maximum value for numerical stability
        return exp_x / np.sum(exp_x)

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def LReLU(self, z):
        ''' LReLU activation function for hidden layer
        z - input
        '''
        np.where(z < 0, self.alpha * z, self.beta * z)
        return z
    
    def LReLU_derivative(self, z):
        ''' Derivative of LReLU activation function for hidden layer
        z - input
        '''
        np.where(z < 0, self.alpha, self.beta)
        return z
