# Note: you are free to organize your code in the way you find most convenient.
# However, make sure that when your main notebook is run, it executes the steps indicated in the assignment.
import numpy as np
from Activations import softmax, sigmoid, LReLU

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
    """
    An artificial neural network for classification problem.
    :param hidden_layer_sizes: list of integers, the ith element represents the number of neurons in the ith hidden layer.
    :param lr: learning rate
    :param activations: list of activation functions to be used in hidden layers
    :param loss_function: loss function to be used
    :param number_of_features: number of features in the dataset
    :param random_state: random seed
    """

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
        # self.hidden_layer_sizes.insert(0, number_of_features)

        # initialise weights using He initialisation
        self.weights = [np.random.normal(loc = 0.0, scale =  2 / (j), size = (i, j))
                        for i, j in zip(hidden_layer_sizes[1:], hidden_layer_sizes[:-1])]
        # initialise biases using random
        self.biases = [np.random.randn(x, 1) for x in hidden_layer_sizes[1:]]

        self.batch_size = batch_size
        self.random_state = random_state

        self.epoch_score = [[], []]


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
        
        # update weights and biases for the whole batch
        self.weights = [weight - (lr / len(small_batch)) * weight_gradient for weight, weight_gradient in zip(self.weights, gradients_of_weights)]
        self.biases = [bias - (lr / len(small_batch)) * bias_gradient for bias, bias_gradient in zip(self.biases, gradients_of_biases)]
        

    def backpropagation(self, x, y):
        """
        Apply backpropagation on the network to acquire the gradients for bias and weight in the tested datapoint
        :param x: feature of the single datapoint
        :param y: label of the datapoint
        :return: list of gradients for weights
        :return: list of gradients for biases
        """
        bias_gradient = [np.zeros(bias.shape) for bias in self.biases]
        weight_gradient = [np.zeros(weight.shape) for weight in self.weights]

        # Alphas - neuron layer *after* applying activation function
        # Zetas - neuron layer *before* applying activation function
        alphas, zetas = self.calculate_az(x)

        delta = self.loss_function.get_difference(alphas[-1], y, zetas[-1])

        bias_gradient[-1] = delta
        weight_gradient[-1] = np.dot(delta, alphas[-2].T)

        # Go through the network backwards
        for i in reversed(range(len(self.hidden_layer_sizes) - 2)):
            weights = self.weights[i + 1]
            delta = np.dot(weights.T, delta) * LReLU(zetas[i], derivative=True)

            # Update gradients
            bias_gradient[i] = delta
            weight_gradient[i] = np.dot(delta, alphas[i].T)

        return weight_gradient, bias_gradient

    def calculate_az(self, x):
        alphas = [x]
        zetas = []
        zeta = x

        # Feed forward the network to calculate alphas and zetas
        for weight, bias in zip(self.weights[:-1], self.biases[:-1]):
            # Calculate the neuron layer weight @ zeta_before + bias
            zeta = np.dot(weight, zeta) + bias
            zetas.append(zeta)

            # Apply activation function to the neuron layer
            alpha = LReLU(zeta)
            alphas.append(alpha)

        # Calculate the last layer
        weight, bias = self.weights[-1], self.biases[-1]

        # Calculate the output layer weight @ zeta_before + bias
        zeta = np.dot(weight, zeta) + bias
        zetas.append(zeta)

        # Apply activation function to the output layer
        alpha = softmax(zeta)
        alphas.append(alpha)

        return alphas, zetas

    def fit(self, X, y, number_of_epochs):
        """
        Method to train the neural network by learning the weights through
        stochastic gradient descent and backpropagation.
        Keeps writing the score for each epoch
        :param X: 
        :param number_of_eopchs: 
        :param mini_batch_size: 
        """
        train_data, test_data = split_dataset(X, y, 0.2)
        n = len(train_data)

        for i in range(number_of_epochs):
            np.random.shuffle(train_data)
            mini_batches = self.create_mini_batches(train_data, self.batch_size, n)

            for mini_batch in mini_batches:
                self.perform_batch_updates(mini_batch, self.lr)

            print("Epoch ", str(i + 1), " done.")
            score = self.score(test_data)
            print("Score (accuracy) for this epoch = ", score)

    def fit_train_val(self, train_data, validation_data, number_of_epochs):
        """
        Method to train the neural network by learning the weights through
        stochastic gradient descent and backpropagation.
        Keeps writing the score for each epoch
        :param X:
        :param number_of_eopchs:
        :param mini_batch_size:
        """
        n = len(train_data)

        for i in range(number_of_epochs):
            np.random.shuffle(train_data)
            mini_batches = self.create_mini_batches(train_data, self.batch_size, n)

            for mini_batch in mini_batches:
                self.perform_batch_updates(mini_batch, self.lr)

            print("Epoch ", str(i + 1), " done.")
            score_val = self.score(validation_data)
            self.epoch_score[0].append(score_val)
            score_train = self.score(train_data)
            self.epoch_score[1].append(score_train)
            print("Score (accuracy) for this epoch on train: ", score_train, ", on validation: ", score_val)

    def only_fit(self, data, number_of_epochs):
        """
            Method to train the neural network by learning the weights through
            stochastic gradient descent and backpropagation.
            :param X:
            :param number_of_eopchs:
            :param mini_batch_size:
        """
        n = len(data)
        for i in range(number_of_epochs):
            np.random.shuffle(data)
            mini_batches = self.create_mini_batches(data, self.batch_size, n)

            for mini_batch in mini_batches:
                self.perform_batch_updates(mini_batch, self.lr)

            # print("Epoch ", str(i + 1), " done.")

    def create_mini_batches(self, X, batch_size, n):
        batches = []
        for i in range(0, n, batch_size):
            batch = X[i : i + batch_size]
            batches.append(batch)
        return batches

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

    def score(self, test_data):
        all = len(test_data)
        corr = 0
        for x, y in test_data:
            output = self.forward_propagate(x)
            corr = corr + 1 if np.argmax(output) == np.argmax(y) else corr
        accuracy = corr / all
        return accuracy

def kfold_cross_validation(X, y, hidden_layer_sizes, learning_rate, loss_function, k=4, num_of_features = 10, random_state = 42, batch_size = 32, num_of_epochs = 10):
    """
    Performs k-fold cross-validation on the input data using the specified model.
    Returns the average accuracy score across all folds.
    """
    num_of_trainings = 10

    data = list(zip(X, y))
    np.random.shuffle(data)

    n = len(data)
    data = np.array(data, dtype='object')
    fold_size = n // k
    scores = []
    for i in range(k):
        print("cross-validation ", i + 1, " / ", k, " step")
        start = i * fold_size
        end = (i + 1) * fold_size
        val_indices = range(start, end)
        train_indices = list(set(range(len(data))) - set(val_indices))
        data_train = data[train_indices]
        data_test = data[val_indices]
        avg_score = []
        for j in range(num_of_trainings):
            print(" - ", j + 1, " / ", num_of_trainings, " iterations.")
            model = ANN(hidden_layer_sizes, learning_rate, loss_function, num_of_features, random_state, batch_size)
            model.only_fit(data_train, num_of_epochs)
            score = model.score(data_test)
            print("Score: ", score)
            avg_score.append(score)
        scores.append(avg_score)
    return scores