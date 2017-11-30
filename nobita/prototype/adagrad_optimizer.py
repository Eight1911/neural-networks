"""
NOTE to self: 
    implement variational autoencoders
    implement self organizing maps

    implement cross validation
    implement L1 regularization

"""

from numpy import dot, outer, zeros, sqrt, square, einsum
from numpy.random import rand, shuffle, normal, seed
from numpy.core.umath_tests import inner1d


import optimizer_functions.optimizer_function
import regularizers
import activations
import neural_nets
import optimizer
import losses




class adagrad_optimizer(optimizer.optimizer):
    # It is not the strongest of the species that survive,
    # nor the most intelligent,
    # but the one most responsive to change.
    """
    adaptive gradient descent ADAGRAD
    """

    def __init__(self, network, l_rate = 0.01, loss = losses.euclidean, epsilon = 1e-4):
        optimizer.optimizer.__init__(self, network, l_rate, loss)
        part_sums = [zeros(n) + epsilon for n in network.arch[1:]]
        self.update_seq = list(zip(network.weights, network.biases, part_sums))
        self.part_sums  = part_sums
        self.epsilon    = epsilon


    @property
    def adagrad_unwind(self):
        epsilon = self.epsilon
        return epsilon


    def __call__(self, stream, iterations):


        def update(update_seq, vecs, deltas, l_rate):
            stream = zip(deltas, vecs, update_seq)
            for delta, vec, (weight, bias, part_sum) in stream:
                gradient = outer(delta, vec)
                part_sum += einsum('ji,ji->j', gradient, gradient) + square(delta)
                # inner1d(gradient, gradient) + square(delta)
                part_inv = 1 / sqrt(part_sum)
                weight += l_rate * (part_inv * gradient.T).T
                if bias is not None:
                    bias += l_rate * part_inv * delta


        def dropout_iterate(self, stream, iterations):
            drop_pairs, batch_size, set_dropout = regularizers.dropout_unwind(self)
            weights, vecs, deltas, l_rate, d_rate, loss = self.unwind
            clipnorm, maxnorm_2 = regularizers.clipnorm_unwind(self)
            push_forth, pull_back = self.compile_functions
            front_arch, back_arch = self.compile_sequences
            update_seq = self.update_seq
            d_rate = 1 - l_rate * d_rate
            vec = vecs[0]
            base_iteration = (iterations + batch_size) // batch_size
            for i in range(base_iteration):
                set_dropout(drop_pairs)
                for batch in range(batch_size):
                    vec[:], target = next(stream)
                    push_forth(vecs, front_arch, d_rate)
                    error = loss(target, vecs[-1])
                    pull_back(deltas, vecs, error, back_arch)
                    update(update_seq, vecs, deltas, l_rate)
                    if maxnorm_2: clipnorm(weights, maxnorm_2)
                if i % 100 == 0: print(dot(error, error), flush = True)


        def iterate(self, stream, iterations):
            weights, vecs, deltas, l_rate, d_rate, loss = self.unwind
            clipnorm, maxnorm_2 = regularizers.clipnorm_unwind(self)
            push_forth, pull_back = self.compile_functions
            front_arch, back_arch = self.compile_sequences
            update_seq = self.update_seq
            d_rate = 1 - l_rate * d_rate
            vec = vecs[0]
            for i in range(iterations):
                vec[:], target = next(stream)
                push_forth(vecs, front_arch, d_rate)
                error = loss(target, vecs[-1])
                pull_back(deltas, vecs, error, back_arch)
                update(update_seq, vecs, deltas, l_rate)
                if maxnorm_2: clipnorm(weights, maxnorm)
                if i % 1000 == 0: print(dot(error, error), flush = True)


        def main(self, stream, iterations):
            if self.dropout:
                p_vec = self.p_vec
                weights = self.network.weights
                regularizers.scale_weights(weights, p_vec)
                dropout_iterate(self, stream, iterations)
                regularizers.descale_weights(weights, p_vec)
            else:
                iterate(self, stream, iterations)

        return main(self, stream, iterations)

import numpy as np
import matplotlib.pyplot as plt

def stream():
    a = np.linspace(-1, 1.0, 2000)
    b = np.sin(2*a) * 0.4 + 0.5
    li = list((np.array([a]), np.array([b])) for a, b in zip(a, b))
    while True:
        shuffle(li)
        for a, b in li:
            yield a, np.sin(6*a) 

arch = [1, 30, 30, 30, 30, 30, 1]
test = neural_nets.network(arch, bias_vec = True)
trainer = adagrad_optimizer(test, l_rate = 0.5)
# trainer.add_weight_decay(0.00001)
# trainer.add_scaling_noise(0.001)
# trainer.add_gaussian_noise(0.001)
# trainer.add_dropout(0.90)
# trainer.add_maxnorm(17)

for i in range(4):
    trainer(stream(), 2000)
    data = [np.array([a]) for a in np.linspace(-1, 1, 2000)]
    a = np.linspace(-1, 1.0, 2000)
    plt.scatter(a, np.sin(6*a), s = 0.01)
    plt.plot(a, test.predict_all(data))
plt.show()


#"""