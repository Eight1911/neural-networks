
# we make mistakes,
# but mistakes also make us.


import numpy as np
import numpy.random as rand
import time


def linear(target, prediction):
    return (target > prediction).astype(int) * 2 - 1


def euclidean(target, prediction):
    return target - prediction


def log_prob(target, prediction):
    pass

def kl_divergence(target, prediction):
    pass

    
