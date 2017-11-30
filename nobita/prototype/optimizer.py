from numpy import dot, outer, zeros, float128
from numpy.random import rand
import numpy as np


import optimizer_functions.optimizer_function
import input_stream
import activations 
import neural_nets
import losses

print("Fourier at Your Service!")

class optimizer:
    # we were all taught to hurry
    # but we were never taught to wait

    def __init__(self, network, l_rate = 0.01, loss = losses.euclidean):
        self.n = len(network.weights)
        self.network = network
        self.l_rate = l_rate
        self.loss = loss

        # regularization methods
        self.maxnorm = False
        self.dropout = False
        self.weight_decay = False
        self.scaling_noise = False
        self.gaussian_noise = False

        self.vecs   = [zeros(n).astype(float128) for n in network.arch]
        self.deltas = [zeros(n).astype(float128) for n in network.arch]

    @property
    def unwind(self):
        weights = self.network.weights
        d_rate = self.weight_decay
        l_rate = self.l_rate
        deltas = self.deltas
        loss = self.loss
        vecs = self.vecs
        return weights, vecs, deltas, l_rate, d_rate, loss


    def add_dropout(self, p_vec = 0.95, batch_size = 60):
        if isinstance(p_vec, (float, int)):
            p_vec = [p_vec for i in range(self.n)]
        drop_vecs = [zeros(n) for n in self.network.arch[:-1]]
        self.drop_pairs = list(zip(drop_vecs, p_vec))
        self.batch_size = batch_size
        self.drop_vecs = drop_vecs
        self.dropout = True
        self.p_vec = p_vec
        

    def add_scaling_noise(self, scale_sigmas = 0.05):
        network = self.network
        if isinstance(scale_sigmas, (float, int)):
            scale_sigmas = [scale_sigmas for i in range(len(network.weights))]
        scale_vecs = [zeros(n) for n in network.arch[:-1]]
        self.scale_pairs = list(zip(scale_vecs, scale_sigmas))
        self.scale_sigmas = scale_sigmas
        self.scale_vecs = scale_vecs
        self.scaling_noise = True


    def add_gaussian_noise(self, gauss_sigmas = 0.01):
        if isinstance(gauss_sigmas, (float, int)):
            gauss_sigmas = [gauss_sigmas for i in range(self.n)]
        self.gauss_sigmas = gauss_sigmas
        self.gaussian_noise = True

    def add_weight_decay(self, weight_decay = 0.0000001):
        self.weight_decay = weight_decay


    def add_maxnorm(self, maxnorm = 4):
        self.maxnorm = maxnorm


    @property
    def compile_functions(self):
        functor = optimizer_functions.optimizer_function.optimizer_function(self)
        return functor.front_pass, functor.back_prop

    @property
    def compile_sequences(self):
        index, weights, biases, acts, primes = self.network.unwind
        front_arch_seq = [index, acts, weights, biases]
        back_arch_seq = [index, primes, weights]

        if self.dropout:
            front_arch_seq.append(self.drop_vecs)
            back_arch_seq.append(self.drop_vecs)

        if self.gaussian_noise:
            front_arch_seq.append(self.gauss_sigmas)

        if self.scaling_noise:
            front_arch_seq.append(self.scale_sigmas)
            front_arch_seq.append(self.scale_vecs)
            back_arch_seq.append(self.scale_vecs)

        front_arch = list(zip(*front_arch_seq))
        back_arch  = list(zip(*back_arch_seq))
        back_arch.reverse()
        return front_arch, back_arch
