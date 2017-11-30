"""
NOTE to self:

    add softmax

"""

import numpy as np
import numpy.random as rand
import time

def leaky_relu(leak = 0.01):
    # happy families are all alike; 
    # every unhappy family is unhappy in its own way.
    def act(x):
        new = x.copy()
        new[x < 0] *= leak
        return new

    def prime(vec):
        new = (vec > 0).astype(np.float128)
        new[new == 0.0] = leak
        return new

    return act, prime

def relu():

    def act(x):
        return np.maximum(x, 0)

    def prime(vec):
        return (vec > 0)

    return act, prime

def modified_relu(leak = 0.01):

    def act(x):
        return np.maximum(x, 0)

    def prime(vec):
        new = (vec > 0).astype(np.float128)
        new[new == 0.0] = leak
        return new

    return act, prime

def sigmoid():

    def act(x):
        return 1 / (1 + np.exp(-x))

    def prime(vec):
        return vec * (1 - vec)

    return act, prime

def tanh():

    def act(x):
        return np.tanh(x)

    def prime(vec):
        return 1 - vec**2

    return act, prime



def softplus():

    def act(x):
        return np.log(1 + np.exp(x)) # center the function

    def prime(vec):
        e = np.exp(vec)
        return (e - 1)/e
    
    return act, prime


def softplus_centered():

    def act(x):
        return np.log(1 + np.exp(x)) - 0.6931471806 # ln(2) to center the function

    def prime(vec):
        return 1 - 0.5/np.exp(vec)
    
    return act, prime

def linear():

    def act(x):
        return x

    def prime(vec):
        return 1
    
    return act, prime


def softmax():

    def act(x):
        s = np.exp(x)
        return s / sum(s)

    def prime(vec):
        pass
        