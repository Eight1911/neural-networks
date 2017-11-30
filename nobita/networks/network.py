string = """
import numpy import dot, exp, log
import numpy as np

def run(network):

"""



from numpy import dot, sqrt
from numpy.random import randn
import numpy as np


from functions import activations


def lzip(*seq):
    return list(zip(*seq))

class network:
    # The whole is more than the sum of its parts.

    def __init__(self, arch, activators = None, bias = True):
        self.n = n = len(arch) - 1

        if bias in [True, False]:
            bias = [bias] * n

        if activators is None:
            activators = n*[activations.softplus()]

        if not isinstance(activators, (list, tuple)):
            activators = [activators] * n

        self.acts, self.primes = lzip(*activators)
        self.dims = lzip(arch, arch[1:])
        self.funs = activators
        self.bias = bias
        self.arch = arch
        self.initialize()


    def initialize(self):
        # xavier weight initialization
        ws = [0] * self.n
        bs = self.bias.copy()
        for i, (col, row) in enumerate(self.dims):
            factor = 2 / sqrt(row + col)
            ws[i] = factor * randn(col, row)
            bs[i] = bs[i] = factor * randn(row) if bs[i] else None

        self.ws, self.bs = ws, bs
        self.predictors = lzip(self.acts, ws, bs)

    def __call__(self, data):
        for a, w, b in zip(self.acts, self.ws, self.bs):
            if b is not None:
                data = a(dot(data, w) + b)
            else:
                data = a(dot(data, w))
        return data
