
import matplotlib.pyplot as plt
from numpy import dot, float128, sqrt
from numpy.random import randn


import activations


class network:
    # The whole is more than the sum of its parts.

    def __init__(self, arch, activators = None, bias_vec = True):
        self.n = n = len(arch) - 1

        if bias_vec in [True, False]:
            bias_vec = [bias_vec for i in range(n)]

        if activators is None:
            activators = [activations.modified_relu() for i in range(n-1)] + [activations.linear()]

        if not isinstance(activators, list):
            activators = [activators for i in range(n)]


        self.acts, self.primes = list(zip(*activators))
        self.activators = list(zip(*activators))
        self.dims = list(zip(arch[1:], arch))
        self.bias_vec = bias_vec
        self.arch = arch
        self.initialize()


    def initialize(self):
        # xavier weight initialization
        weights = [0] * len(self.dims)
        biases =  self.bias_vec.copy()

        for index, (col, row) in enumerate(self.dims):
            factor = 2 / sqrt(row + col)
            weights[index] = factor * randn(col, row).astype(float128)

            if biases[index]:
                biases[index] = factor * randn(col).astype(float128)
            else:
                biases[index] = None

        self.weights, self.biases = weights, biases
        self.predictor_ensemble = list(zip(self.acts, weights, biases))


    def predict(self, vec):
        for act, weight, bias in self.predictor_ensemble:
            vec = act(dot(weight, vec)) if bias is None else act(dot(weight, vec) + bias)
        return vec

    def predict_all(self, vecs):
        predict = self.predict
        return [predict(vec) for vec in vecs]


    def plot(self, inputs):
        import matplotlib.pyplot as plt
        inputs = [array([vec]) for vec in inputs]
        predictions = [float(i) for i in self.predict_all(inputs)]
        plt.plot(inputs, predictions)
        plt.show()

    @property
    def unwind(self):
        index = list(range(1, self.n + 1))
        weights = self.weights
        biases = self.biases
        primes = self.primes
        acts = self.acts
        return index, weights, biases, acts, primes


    def count_fires(self, inputs):
        # counts the number of times each node fires
        # when given the inputs
        raise NotImplementedError

    def simplified_architecture(self, inputs, threshold = 0.0001):
        # delete all nodes that fire
        # less than threshold proprtion
        # of the time when given the data set
        raise NotImplementedError # yet
