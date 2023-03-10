# Note: you are free to organize your code in the way you find most convenient.
# However, make sure that when your main notebook is run, it executes the steps indicated in the assignment.
import numpy as np
from Activations import step_function, softmax, sigmoid, LReLU
# from LossFunctions import *

# split data into train and test sets

def split_dataset(X, Y, test_size):

    # compute sizes of sets
    assert 0 <= test_size <= 1
    data = list(zip(X, Y))    
    np.random.shuffle(data)

    index = int(len(data) * test_size)
    return data[index:], data[:index]

class LogLikelihood:

    def get_difference(a, y, z):
        return a - y

    def get_cost(data, network):
        total = 0
        for x, y in data:
            output = network.forward_propagate(x)
            total += np.dot(y.reshape(y.shape[0]), np.log(output).reshape(output.shape[0]))
        return - (total / len(data))

class QuadraticCost:

    def get_difference(a, y, z):
        return (a - y) * sigmoid(z, derivative=True)

    def get_cost(data, network):
        all = 0
        for x, y in data:
            output = network.forward_propagate(x)
            all += 0.5 * (np.linalg.norm(output - y) ** 2)
        return -(all / len(data))

class ANN:
    '''
    An artificial neural network for classification problem.
    :param hidden_layer_sizes: list of integers, the ith element represents the number of neurons in the ith hidden layer.
    :param lr: learning rate
    :param activations: list of activation functions to be used in hidden layers
    :param loss_function: loss function to be used
    :param number_of_features: number of features in the dataset
    :param random_state: random seed
    '''

    def __init__(self, hidden_layer_sizes: list, lr: float, loss_function: str, number_of_features: int = 10, random_state: int = 42, batch_size: int = 32):
        self.lr = lr
        if loss_function == 'square':
            self.loss_function = QuadraticCost
        elif loss_function == 'log':
            self.loss_function = LogLikelihood
        else:
            print("Function not recognized. Abort.")
            return

        # creating a list of layer sizes
        self.hidden_layer_sizes = hidden_layer_sizes

        self.number_of_features = number_of_features

        # prepend the number of features to the list of hidden layer sizes
        self.hidden_layer_sizes.insert(0, number_of_features)

        # initialise weights using He initialisation
        self.weights = [np.random.normal(loc = 0.0, scale =  2 / (j), size = (i, j))
                        for i, j in zip(hidden_layer_sizes[1:], hidden_layer_sizes[:-1])]
        # initialise biases using random
        self.biases = [np.random.randn(x, 1) for x in hidden_layer_sizes[1:]]

        self.batch_size = batch_size
        self.random_state = random_state

    

    def perform_batch_updates(self, small_batch, lr):
        """
        Passes a small batch through the network and updates the weights and biases
        :param mini_batch: random subset of the whole training set
        :param learning_rate: gradient descent learning rate
        :return: None
        """

        gradients_of_biases = []
        gradients_of_weights = []

        for b in self.biases:
            gradients_of_biases.append(np.zeros(b.shape))
        for w in self.weights:
            gradients_of_weights.append(np.zeros(w.shape))

        for feature, label in small_batch:
            # go back through the network and compute the gradient of the cost function
            point_weight_gradient, point_bias_gradient = self.backpropagation(feature, label)
            
            gradients_of_weights = [wg + pwg for wg, pwg in zip(gradients_of_weights, point_weight_gradient)]
            gradients_of_biases = [bg + pbg for bg, pbg in zip(gradients_of_biases, point_bias_gradient)]
        
        #update weights and biases for the whole batch
        self.weights = [weight - (lr / len(small_batch)) * weight_gradient for weight, weight_gradient in zip(self.weights, gradients_of_weights)]
        self.biases = [bias - (lr / len(small_batch)) * bias_gradient for bias, bias_gradient in zip(self.biases, gradients_of_biases)]
        

    def backpropagation(self, x, y):
        """
        Method that applies backpropagation using x as input and returns the
        gradient of the cost function.
        :param x: input point
        :param y: label of x
        :return: gradient of cost function
        """
        bias_gradients = [np.zeros(bias.shape) for bias in self.biases]
        weight_gradients = [np.zeros(weight.shape) for weight in self.weights]

        # Alphas - neuron layer *after* applying activation function
        # Zetas - neuron layer *before* applying activation function
        alphas, zetas = self.calculate_az(x)

        delta = self.loss_function.get_difference(alphas[-1], y, zetas[-1])

        bias_gradients[-1] = delta
        weight_gradients[-1] = np.dot(delta, alphas[-2].T)

        # Go through the network backwards
        for i in reversed(range(len(self.hidden_layer_sizes) - 2)):

            weights = self.weights[i + 1]
            delta = np.dot(weights.T, delta) * LReLU(zetas[i], derivative=True)

            bias_gradients[i] = delta
            weight_gradients[i] = np.dot(delta, alphas[i].T)

        return weight_gradients, bias_gradients

    def calculate_az(self, x):
        alphas = [x]
        zetas = []
        zeta = x

        # Feed forward the network to calculate alphas and zetas
        for weight, bias in zip(self.weights[:-1], self.biases[:-1]):
            zeta = np.dot(weight, zeta) + bias
            zetas.append(zeta)
            alpha = LReLU(zeta)
            alphas.append(alpha)

        weight, bias = self.weights[-1], self.biases[-1]
        zeta = np.dot(weight, zeta) + bias
        zetas.append(zeta)
        alpha = softmax(zeta)
        alphas.append(alpha)
        return alphas, zetas

    def fit(self, X, number_of_epochs, mini_batch_size):
        """
        Method to train the neural network by learning the weights through
        stochastic gradient descent and backpropagation.
        :param X: 
        :param number_of_eopchs: 
        :param mini_batch_size: 
        """
        n = len(X)

        for i in range(number_of_epochs):
            np.random.shuffle(X)
            mini_batches = [X[j:j + mini_batch_size] for j in range(0, n, mini_batch_size)]

            for mini_batch in mini_batches:
                self.perform_batch_updates(mini_batch, self.lr)

            print("Epoch ", str(i + 1), " done.")

    def forward_propagate(self, layer):
        """
        Propagate the input and get the output layer
        :param layer: input layer
        :return: output layer
        """
        # Propagate the input through the layers up to the last one
        for weight, bias in zip(self.weights[:-1], self.biases[:-1]):
            # layer = sigmoid(np.dot(weight, layer) + bias)
            layer = LReLU(np.dot(weight, layer) + bias)

        # Use softmax function for the last layer
        weight, bias = self.weights[-1], self.biases[-1]
        layer = softmax(np.dot(weight, layer) + bias)
        return layer

