import numpy as np

def step_function(z):
    '''
    Step activation function for output layer
    z - input
    '''
    return np.where(z >= 0, 1, 0)

def softmax(z):
    '''
    Softmax activation function for output layer
    z - input
    '''

    r = z
    exp_x = np.exp(r - np.max(r))  # subtract the maximum value for numerical stability
    return exp_x / np.sum(exp_x)

def tanh(z, derivative=False):
    '''
    Tanh activation function for hidden layer
    z - input
    '''
    if not derivative:
        return np.tanh(z)
    else:
        return 1 - np.tanh(z)**2

def sigmoid(z, derivative=False):
    '''
    Sigmoid activation function for hidden layer
    z - input
    '''
    res = 1 / (1 + np.exp(-z))
    if not derivative:
        return res
    else:
        return res * (1 - res)
     

def LReLU(z, derivative=False):
    '''
    LReLU activation function for hidden layer
    z - input
    '''
    alpha = 0.1
    beta = 1
    if not derivative:
        return np.maximum(alpha*z, beta*z)
    else:
        result = np.copy(z)
        result[result < 0] *= alpha
        return result
        # dx = np.ones_like(z)
        # Alpha * x (?)
        # return np.maximum(alpha * z, z)
        # return dx
