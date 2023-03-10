# Note: you are free to organize your code in the way you find most convenient.
# However, make sure that when your main notebook is run, it executes the steps indicated in the assignment.
import numpy as np
from Activations import step_function, softmax, sigmoid, LReLU
from LossFunctions import QuadraticCost, LogLikelihood

# split data into train and test sets

def split_dataset(X, Y, test_size):

    # compute sizes of sets
    assert 0 <= test_size <= 1
    data = list(zip(X, Y))    
    np.random.shuffle(data)

    index = int(len(data) * test_size)
    return data[index:], data[:index]

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
        '''
        initialize weights using He initialization. 
        '''
        
        self.lr = lr
        if loss_function == 'square':
            self.loss_function = QuadraticCost
        elif loss_function == 'log':
            self.loss_function = LogLikelihood

        # creating a list of layer sizes
        self.hidden_layer_sizes = hidden_layer_sizes
        self.number_of_features = number_of_features
        # prepend the number of features to the list of hidden layer sizes
        self.hidden_layer_sizes.insert(0, number_of_features)

        #innitialize weights and biases using He initialization
        self.weights = [np.random.normal(loc = 0.0, scale =  2 / (j), size = (i, j))
                        for i, j in zip(hidden_layer_sizes[1:], hidden_layer_sizes[:-1])]
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

        alphas = [x]
        zetas = []
        a = alphas[0]
        # Feed point forward and store all activations and outputs
        for w, b in zip(self.weights[:-1], self.biases[:-1]):
            z = np.dot(w, a) + b
            zetas.append(z)
            a = sigmoid(z)
            alphas.append(a)

        w, b = self.weights[-1], self.biases[-1]
        z = np.dot(w, a) + b
        zetas.append(z)
        a = softmax(z)
        alphas.append(a)

        # Calculate cost (delta) for output layer and use it to calculate
        # gradients of the output layer
        delta = self.loss_function.get_difference(alphas[-1], y, zetas[-1])

        bias_gradients[-1] = delta

        weight_gradients[-1] = np.dot(delta, alphas[-2].T)

        # Move back through the network calculating gradients by updating
        # delta

        for i in reversed(range(len(self.hidden_layer_sizes) - 2)):

            weights = self.weights[i + 1]
            delta = np.dot(weights.T, delta) * sigmoid(zetas[i], derivative=True)

            bias_gradients[i] = delta
            weight_gradients[i] = np.dot(delta, alphas[i].T)

        return weight_gradients, bias_gradients
    
    # def fit(self, X, number_of_eopchs, mini_batch_size, learning_rate, validation_data=None):
    def fit(self, X, number_of_eopchs, mini_batch_size, learning_rate):
        """
        Method to train the neural network by learning the weights through
        stochastic gradient descent and backpropagation.
        :param X: 
        :param number_of_eopchs: 
        :param mini_batch_size: 
        """
        n = len(X)

        for i in range(number_of_eopchs):
            np.random.shuffle(X)
            mini_batches = [X[j:j + mini_batch_size] for j in range(0, n, mini_batch_size)]

            for mini_batch in mini_batches:
                self.perform_batch_updates(mini_batch, learning_rate)

            print(f"Epoch {i + 1}")

    def forward_propagate(self, layer):
        """
        Forward propagate the input through the network and get the output
        :param x: The input to the ann
        :return: The output of the ann
        """
        # forward propagate the input through the network for all but the last layer
        for weight, bias in zip(self.weights[:-1], self.biases[:-1]):
            layer = sigmoid(weight @ layer + bias)

        w, b = self.weights[-1], self.biases[-1]
        layer = softmax(np.dot(w, layer) + b)
        return layer
    