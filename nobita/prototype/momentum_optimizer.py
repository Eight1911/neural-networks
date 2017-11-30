"""
NOTE to self:
    implement variational autoencoders
    implement self organizing maps
    implement cross validation
    implement L1 regularization

"""

from numpy import dot, outer, zeros, sqrt
from numpy.random import rand, shuffle, normal, seed


import optimizer_functions.optimizer_function
import regularizers
import activations
import neural_nets
import optimizer
import losses


# seed(0)

class momentum_optimizer(optimizer.optimizer):
    # we have only this moment
    # sparkling like a star in our hand
    # and melting like a snowflake.
    """
    momentum trainer with optional Nesterov's
    acceleration and learning rate decay
    """
    def __init__(self, network, l_rate = 0.01, loss = losses.euclidean, l_decay = 0.000005, moment = 0.95, nesterov = True):
        optimizer.optimizer.__init__(self, network, l_rate, loss)
        weight_momenta = [zeros(weight.shape) for weight in network.weights]
        bias_momenta   = [None if bias is None else zeros(bias.size) for bias in network.biases]
        self.update_seq = list(zip(network.weights, network.biases, weight_momenta, bias_momenta))
        self.l_decay = 1 - l_decay if l_decay else True # learning rate decay
        self.weight_momenta = weight_momenta
        self.bias_momenta = bias_momenta
        self.nesterov = nesterov
        self.moment = moment


    @property
    def momentum_unwind(self):
        l_decay = self.l_decay
        moment = self.moment
        return l_decay, moment


    def __call__(self, stream, iterations):

        def momentum_update(update_seq, vecs, deltas, moment, l_rate):
            stream = zip(deltas, vecs, update_seq)
            for delta, vec, (weight, bias, weight_momentum, bias_momentum) in stream:
                weight_momentum *= moment
                weight_momentum += outer(delta, vec)
                weight += l_rate * weight_momentum
                if bias is not None:
                    bias_momentum *= moment
                    bias_momentum += delta
                    bias += l_rate * bias_momentum


        def nesterov_update(update_seq, vecs, deltas, moment, l_rate):
            stream = zip(deltas, vecs, update_seq)
            for delta, vec, (weight, bias, weight_momentum, bias_momentum) in stream:
                gradient = outer(delta, vec)
                weight_momentum *= moment
                nesterov_weight = gradient + moment * (gradient + weight_momentum)
                weight_momentum += gradient
                weight += l_rate * nesterov_weight
                if bias is not None:
                    bias_momentum *= moment
                    nesterov_bias = delta + moment * (delta + bias_momentum)
                    bias_momentum += delta
                    bias += l_rate * nesterov_bias


        def dropout_iterate(self, stream, iterations):
            drop_pairs, batch_size, set_dropout = regularizers.dropout_unwind(self)
            weights, vecs, deltas, l_rate, d_rate, loss = self.unwind
            update = nesterov_update if self.nesterov else momentum_update
            clipnorm, maxnorm_2 = regularizers.clipnorm_unwind(self)
            push_forth, pull_back = self.compile_functions
            front_arch, back_arch = self.compile_sequences
            update_seq = self.update_seq
            l_decay, moment = self.momentum_unwind

            vec = vecs[0]
            l_rate *= 1 - moment
            base_iteration = (iterations + batch_size) // batch_size
            for i in range(base_iteration):
                set_dropout(drop_pairs)
                for batch in range(batch_size):
                    l_rate *= l_decay
                    d_rate *= l_decay
                    vec[:], target = next(stream)
                    push_forth(vecs, front_arch, 1 - d_rate)
                    error = loss(target, vecs[-1])
                    pull_back(deltas, vecs, error, back_arch)
                    update(update_seq, vecs, deltas, moment, l_rate)
                    if maxnorm_2: clipnorm(weights, maxnorm_2)
                if i % 10 == 0: print(dot(error, error), flush = True)


        def iterate(self, stream, iterations):
            weights, vecs, deltas, l_rate, d_rate, loss = self.unwind
            update = nesterov_update if self.nesterov else momentum_update
            clipnorm, maxnorm_2 = regularizers.clipnorm_unwind(self)
            push_forth, pull_back = self.compile_functions
            front_arch, back_arch = self.compile_sequences
            update_seq = self.update_seq
            l_decay, moment = self.momentum_unwind
            vec = vecs[0]
            l_rate *= 1 - moment
            for i in range(iterations):
                l_rate *= l_decay
                d_rate *= l_decay
                vec[:], target = next(stream)
                push_forth(vecs, front_arch, 1 - d_rate)
                error = loss(target, vecs[-1])
                pull_back(deltas, vecs, error, back_arch)
                update(update_seq, vecs, deltas, moment, l_rate)
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
            yield a, np.sin(13*a)

arch = [1] + [10] * 10 + [1]
test = neural_nets.network(arch, bias_vec = True)
trainer = momentum_optimizer(test, l_rate = 0.005, l_decay = 0.00005, moment = 0.1, nesterov = True)
trainer.add_weight_decay(0.00001)
# trainer.add_scaling_noise(0.001)
# trainer.add_gaussian_noise(0.001)
# trainer.add_dropout(0.99)
# trainer.add_maxnorm(17)

for i in range(4):
    trainer(stream(), 9000)
    data = [np.array([a]) for a in np.linspace(-1, 1, 2000)]
    a = np.linspace(-1, 1.0, 2000)
    plt.scatter(a, np.sin(13*a), s = 0.01)
    plt.plot(a, test.predict_all(data))
plt.show()
