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
        e_z = np.exp(z - np.max(z))
        return e_z / e_z.sum(axis=-1, keepdims=True)

    def LReLU(self, z, der = False):
        ''' LReLU activation function for hidden layer
        z - input
        '''
        if(not der):
            np.where(z < 0, self.alpha * z, self.beta * z)
        else:
            np.where(z < 0, self.alpha, self.beta)
        return z
    
    # def LReLU_derivative(self, z):
    #     ''' Derivative of LReLU activation function for hidden layer
    #     z - input
    #     '''
    #     np.where(z < 0, self.alpha, self.beta)
    #     return z
