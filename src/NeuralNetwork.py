# Note: you are free to organize your code in the way you find most convenient.
# However, make sure that when your main notebook is run, it executes the steps indicated in the assignment.
import numpy as np
from Activations import step_function, softmax, sigmoid, LReLU

def train_test_split(X, Y, test_size):

    # split into train and test sets
    assert 0 < test_size < 1, "test_size must be between 0 and 1"
    train_size = 1 - test_size
    train_size = int(train_size * X.shape[0])
    test_size = X.shape[0] - train_size

    # shuffle indexes to get random division into sets
    idx = np.random.permutation(X.shape[0])

    # assign indexes of sets
    train_idx, test_idx = idx[:train_size], idx[train_size : train_size + test_size]
    return X[train_idx,:], X[test_idx,:], Y[train_idx,], Y[test_idx,]


class Loss:
    def categorical_cross_entropy(self, y_true, y_pred):
        '''
        y_true - correct label
        y_pred - label predicted by a model
        '''
        return -np.sum(y_true * np.log(y_pred + 10**-100))

    def squared_error(self, y_true, y_pred, der = False):
        '''
        y_true - correct label
        y_pred - label predicted by a model
        '''
        if not der:
            return np.sum((y_true - y_pred)**2)
        return 2 * (y_pred - y_true)

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

    def __init__(self, hidden_layer_sizes: list, lr: float, activations: list, loss_function: Loss, number_of_features: int = 10, random_state: int = 42, batch_size: int = 32):
        '''
        initialize weights using He initialization. These weights include the biases in them as well.
        '''
        self.weights = [np.random.normal(loc = 0.0, scale =  2 / (j), size = (i, j))
                        for i, j in zip(hidden_layer_sizes, [number_of_features] + hidden_layer_sizes[:-1])]
        self.biases = [np.random.randn(x, 1) for x in hidden_layer_sizes]
        self.lr = lr
        self.loss_function = loss_function
        self.activations = activations
        self.random_state = random_state
        self.hidden_layer_sizes = hidden_layer_sizes
        self.number_of_features = number_of_features
        self.batch_size = batch_size


    def fit(self, features: np.array, targets: np.array, num_of_epochs: int):
        x_train, x_test, x_val, y_train, y_test, y_val = train_test_val_split(features, targets, 0.6, 0.2)
        print("Y: ", y_test.shape)
        print("X: ", x_test.shape)
        self.train(x_train, y_train, num_of_epochs, self.batch_size)
        accuracy = self.evaluate(x_test, y_test)
        print("Accuracy: ", accuracy)

    def evaluate(self, x_test, y_test):
        i = 0
        score = 0
        for x in x_test:
            pred = self.predict(x)
            print(pred)
            if(pred == y_test[i]):
                score += 1
            i += 1
        return score / len(y_test)

    def feed_forward(self, x):
        """
        Method that calculates the output for each layer in the network.
        :param x: training input.
        :return: alphas - list of neuron values after applying activation function.
        :return: zetas - list of neuron values before applying activation function.
        """
        alphas = []
        zetas = []

        alphas.append(x.reshape(-1, 1))
        X = x
        for W_i, b_i, activation in zip(self.weights, self.biases, self.activations):
            # Calculate the layer and save it in the zetas
            X = np.dot(W_i, X).reshape(-1, 1)
            X += b_i

            zetas.append(X)

            # Calculate the activation function for that layer and save it in the alphas
            X = activation(X)
            alphas.append(X)

        return alphas, zetas

    def predict(self, X: np.array):
        for W_i, b_i, activation in zip(self.weights, self.biases, self.activations):
            X = np.dot(W_i, X).reshape(-1, 1)
            X += b_i
            X = activation(X)

        # chose a label with the highest probability
        print(X)
        return X.argmax(axis=1) + 1

    def train(self, X, y, num_of_epochs, batch_size):
        for i in range(num_of_epochs):
            batches = create_mini_batches(X, y, batch_size)

            for batch in batches:
                self.update_weights_with_batch(batch)

            print("Epoch " + str(i + 1) + " done out of " + str(num_of_epochs))

    def back_propagation(self, X, y):
        '''
        Method that implements back propagation algorithm and returns the gradient of the cost function.
        :param X: input data
        :param y: correct label
        :return: gradient
        '''


        weights_gradients = [np.zeros(w.shape) for w in self.weights]
        biases_gradients = [np.zeros(b.shape) for b in self.biases]

        applied_neuron_values, neuron_values = self.feed_forward(X)


        delta = self.loss_function(y, applied_neuron_values[-1], der = True)

        biases_gradients[-1] = delta
        weights_gradients[-1] = np.dot(delta, neuron_values[-2].T)


        # Moving backwards through the layers
        for i in reversed(range(len(self.hidden_layer_sizes) - 1)):

            weights = self.weights[i + 1]
            delta = np.dot(weights.T, delta) * self.activations[i](neuron_values[i])
            biases_gradients[i] = delta
            weights_gradients[i] = np.dot(delta, applied_neuron_values[i].reshape(1, -1))

        return weights_gradients, biases_gradients

    def update_weights_with_batch(self, batch):
        bias_gradients = [np.zeros((x, 1)) for x in self.hidden_layer_sizes]
        weight_gradients = [np.zeros(w.shape) for w in self.weights]
            # [np.zeros((i, j)) for i, j in zip([self.number_of_features] + self.hidden_layer_sizes[:-1], self.hidden_layer_sizes)]

        for x, y in batch:
            datapoint_weight_gradient, datapoint_bias_gradient = self.back_propagation(x, y)
            bias_gradients = [bg + pbg for bg, pbg in zip(bias_gradients, datapoint_bias_gradient)]
            weight_gradients = [wg + pwg for wg, pwg in zip(weight_gradients, datapoint_weight_gradient)]

        self.biases = [b - (bg * self.lr / self.batch_size) for b, bg in zip(self.biases, bias_gradients)]
        self.weights = [w - (wg * self.lr / self.batch_size) for w, wg in zip(self.weights, weight_gradients)]

def create_mini_batches(X: np.array, y: np.array, batch_size: int):
    """
    Function for dividing the train set into batches of "batch_size".

    :param X: The training features.
    :param y: The labels for the features.
    :param batch_size: Size of the batch.
    :return: List of tuples (X_mini_batch, y_mini_batch) where X and y have "batch_size" length.
    """

    batches = []
    '''Add y values to the corresponding feature values'''
    data = np.hstack((X, y))
    np.random.shuffle(data)
    num_of_batches = data.shape[0] // batch_size

    for i in range(0, num_of_batches):
        '''Take "batch_size" consecutive elements from data'''
        # print("Batch from: ", i * batch_size, " to ", (i + 1) * batch_size)
        batch = data[i * batch_size : (i + 1) * batch_size]
        X_batch = batch[:, :-1]
        # Each y_value is in its own list
        y_batch = batch[:, -1].reshape((-1, 1))
        batches.append(zip(X_batch, y_batch))

    '''If there are some elements left, create new batch for them'''
    if data.shape[0] % batch_size != 0:
        # print("Last batch from: ", num_of_batches * batch_size, " to ", data.shape[0])
        batch = data[num_of_batches * batch_size : data.shape[0]]
        X_batch = batch[:, :-1]
        # Each y_value is in its own list
        y_batch = batch[:, -1].reshape((-1, 1))
        batches.append(zip(X_batch, y_batch))
    return batches


