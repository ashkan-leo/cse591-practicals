# sample_submission.py
import numpy as np

from math import ceil
from scipy.special import expit
sig = expit
tnh = np.tanh

def d_sigmoid(z):
    return np.multiply(sig(z), 1.0 - sig(z))


def sigmoid(z, drv=False):
    if drv:
        return d_sigmoid(z)
    return sig(z)


def d_tanh(z):
    return 1 - np.multiply(tnh(z), tnh(z))


def tanh(z, drv=False):
    if drv:
        return d_tanh(z)
    return tnh(z)


class xor_net(object):
    """
    This is a sample class for miniproject 1.

    Args:
        data: Is a tuple, ``(x,y)``
              ``x`` is a two or one dimensional ndarray ordered such that axis 0 is independent 
              data and data is spread along axis 1. If the array had only one dimension, it implies
              that data is 1D.
              ``y`` is a 1D ndarray it will be of the same length as axis 0 or x.   
                          
    """

    def __init__(self, data, labels, hidden=None, act=None, alpha=0.02, num_iter=3000):
        if hidden is None:
            hidden = [4]
            act = [sigmoid]

        act = [np.identity] + act + [sigmoid]

        if len(hidden) != len(act) - 2:
            raise ValueError('Hidden layer and activation functions must have the same length.')

        # Add the bias term to x
        self.x = np.hstack((np.ones((data.shape[0], 1)), data))
        self.y = np.array(labels)[np.newaxis].T

        self.alpha = alpha
        self.num_iter = num_iter

        # Number of neurons in each layer.
        self.structure = [self.x.shape[1]] + hidden + [1]
        self.l = len(self.structure)
        self.activation = act

        # Initialize the weights and Cost of the network. W[i] & C[i] is the weight and change from layer i to i + 1.
        self.W = []
        self.C = []
        for ix, z in enumerate(self.structure[:-1]):
            o = self.structure[ix + 1]
            self.W.append(np.random.normal(size=(z, o)))
            self.C.append(np.zeros((z, o)))

        # Initialize the activation and delta values for each layer.
        self.A = []
        self.L = []
        for z in self.structure:
            self.A.append(np.ones((z, 1), dtype=float))
            self.L.append(np.zeros((z, 1), dtype=float))

        self.train(zip(self.x, self.y))

    def feed_forward(self, inputs):
        """
            Feed forward algorithm

            Args:
                inputs: First layer value

            Returns:
                numpy.ndarray: last layer value
        """
        if inputs.shape != self.A[0].shape:
            raise ValueError('Wrong input dimension.')

        self.A[0] = inputs
        for l in range(1, self.l):
            self.L[l] = np.dot(self.W[l - 1].T, self.A[l - 1])
            activation = self.activation[l]
            self.A[l] = activation(self.L[l])
        return self.A[-1]

    def back_propagate(self, targets):
        """
            Perform back prop and update weights

            Args:
                targets: labels

            Returns:
                numpy.ndarray: error value
            """
        alpha = self.alpha
        if targets.shape != self.A[-1].shape:
            raise ValueError('Wrong output dimension.')

        # We need delta for all the layers except for the input.
        deltas = [None] * self.l
        # deltas[-1] = sigmoid(self.A[-1], drv=True) * (-(targets - self.A[-1]))

        deltas[-1] = self.activation[-1](self.L[-1], drv=True) * (-(targets - self.A[-1]))
        for l in range(self.l - 2, 0, -1):
            error = np.dot(self.W[l], deltas[l + 1])
            # deltas[l] = np.multiply(sigmoid(self.A[l], drv=True), error)
            activation = self.activation[l]
            deltas[l] = np.multiply(activation(self.L[l], drv=True), error)

        changes = []
        for l in range(self.l - 1):
            change = np.dot(self.A[l], deltas[l + 1].T)
            changes.append(change)
            self.W[l] -= alpha * change + self.C[l]
            self.C[l] = change

        error = 0.5 * (targets - self.A[-1]) ** 2

        return error

    def train(self, data):
        """
            train the network from data

            Args:
                data: tuple of x and y
        """
        num_iter = self.num_iter

        for i in range(num_iter):
            error = 0.0
            for d in data:
                inputs = np.array(d[0])[np.newaxis].T
                targets = np.array(d[1])[np.newaxis].T
                self.feed_forward(inputs)
                error += self.back_propagate(targets)

            # if i % 500 == 0:
            #     print('error %-.5f' % error)

    def get_params(self):
        """ 
        Method that should return the model parameters.

        Returns:
            tuple of numpy.ndarray: (w, b). 

        Notes:
            This code will return an empty list for demonstration purposes. A list of tuples of 
            weights and bias for each layer. Ordering should from input to outputt

        """
        return self.W

    def get_predictions(self, x):
        """
        Method should return the outputs given unseen data

        Args:
            x: array similar to ``x`` in ``data``. Might be of different size.

        Returns:    
            numpy.ndarray: ``y`` which is a 1D array of predictions of the same length as axis 0 of 
                            ``x`` 
        Notes:
            Temporarily returns random numpy array for demonstration purposes.                            
        """
        # Here is where you write a code to evaluate the data and produce predictions.
        x = np.hstack((np.ones((x.shape[0], 1)), x))

        round = lambda x: 1 if x >= 0.5 else 0
        return np.apply_along_axis(lambda r: round(self.feed_forward(r[np.newaxis].T)), axis=1, arr=x).reshape(-1)

        # pred = np.apply_along_axis(lambda r: self.feed_forward(r[np.newaxis].T), axis=1, arr=x).reshape(-1)
        # return np.random.rand(len(pred)) <= pred


class mlnn(xor_net):
    """
    At the moment just inheriting the network above. 
    """

    def __init__(self, data, labels):
        super(mlnn, self).__init__(data, labels, hidden=[100, 50], act=[sigmoid, sigmoid], alpha=0.0002, num_iter=1500)


if __name__ == '__main__':
    pass
