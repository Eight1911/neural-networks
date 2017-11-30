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



# Now the Lord God had planted a garden in the east, in Eden; 
# and there he put the man he had formed. The Lord God made 
# all kinds of trees grow out of the ground—trees that were pleasing 
# to the eye and good for food. In the middle of the garden were 
# the tree of life and the tree of the knowledge of good and evil.



class rmsprop_optimizer(optimizer.optimizer):
    # forgetfulness is a form of freedom.

    # There is no remembrance which time does not obliterate, 
    # nor pain which death does not terminate.
    """
    RMSprop optimizer
    root mean square propagation
    with optional momentum and learning rate decay

    # add gradient noise
    """

    def __init__(self, network, l_rate = 0.2, loss = losses.euclidean, l_decay = 0, moment = 0, sum_decay = 0.001, epsilon = 1e-2):
        optimizer.optimizer.__init__(self, network, l_rate, loss)
        part_sums = [zeros(n) + epsilon for n in network.arch[1:]]
        weight_momenta = [zeros(weight.shape) for weight in network.weights]
        bias_momenta   = [None if bias is None else zeros(bias.size) for bias in network.biases]
        
        self.part_sums = part_sums
        self.sum_decay = sum_decay
        self.epsilon   = epsilon
        self.l_decay   = l_decay
        self.moment    = moment
        self.weight_momenta = weight_momenta
        self.bias_momenta = bias_momenta
        self.update_seq = list(zip( network.weights, 
                                    network.biases, 
                                    part_sums, 
                                    weight_momenta, 
                                    bias_momenta)) # lol syntax troll
        


    @property
    def rmsprop_unwind(self):
        sum_decay = self.sum_decay
        epsilon = self.epsilon
        l_decay = self.l_decay
        moment = self.moment
        return l_decay, epsilon, sum_decay, moment


    def __call__(self, stream, iterations):


        def update(update_seq, vecs, deltas, moment, sum_decay, epsilon, l_rate):
            stream = zip(deltas, vecs, update_seq)
            for delta, vec, (weight, bias, part_sum, weight_momentum, bias_momentum) in stream:
                gradient = outer(delta, vec)
                weight_momentum *= moment
                weight_momentum += gradient
                part_sum *= sum_decay
                # if using float64, then use inner1d instead of einsum -> 50% efficient
                part_sum += einsum('ji,ji->j', gradient, gradient) + square(delta) 
                part_inv = 1 / (sqrt(part_sum) + epsilon)
                weight += l_rate * (part_inv * weight_momentum.T).T
                if bias is not None:
                    bias_momentum *= moment
                    bias_momentum += delta
                    bias += l_rate * part_inv * bias_momentum


        def dropout_iterate(self, stream, iterations):
            # what can dropouts be but billionaires
            # what can graduates be but jobless
            drop_pairs, batch_size, set_dropout = regularizers.dropout_unwind(self)
            weights, vecs, deltas, l_rate, d_rate, loss = self.unwind
            clipnorm, maxnorm_2 = regularizers.clipnorm_unwind(self)
            l_decay, epsilon, sum_decay, moment = self.rmsprop_unwind
            push_forth, pull_back = self.compile_functions
            front_arch, back_arch = self.compile_sequences
            update_seq = self.update_seq

            d_rate = l_rate * d_rate                        # weight decay rate
            l_rate *= 1 - moment
            l_decay = 1 - l_decay if l_decay else True      # learning rate decay
            sum_decay = 1 - sum_decay                       # sum_decay rate

            vec = vecs[0]
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
                    update(update_seq, vecs, deltas, moment, sum_decay, epsilon, l_rate)
                    if maxnorm_2: clipnorm(weights, maxnorm_2)
                if i % 100 == 0: print(dot(error, error), flush = True)


        def iterate(self, stream, iterations):
            weights, vecs, deltas, l_rate, d_rate, loss = self.unwind
            clipnorm, maxnorm_2 = regularizers.clipnorm_unwind(self)
            l_decay, epsilon, sum_decay, moment = self.rmsprop_unwind
            push_forth, pull_back = self.compile_functions
            front_arch, back_arch = self.compile_sequences
            update_seq = self.update_seq

            d_rate = l_rate * d_rate                        # weight decay rate
            l_rate *= 1 - moment
            l_decay = 1 - l_decay if l_decay else True      # learning rate decay
            sum_decay = 1 - sum_decay                       # sum_decay rate

            vec = vecs[0]
            for i in range(iterations):
                l_rate *= l_decay
                d_rate *= l_decay
                vec[:], target = next(stream)
                push_forth(vecs, front_arch, 1 - d_rate)
                error = loss(target, vecs[-1])
                pull_back(deltas, vecs, error, back_arch)
                update(update_seq, vecs, deltas, moment, sum_decay, epsilon, l_rate)
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

#"""
import numpy as np
import matplotlib.pyplot as plt

def stream():
    a = np.linspace(-1, 1.0, 2000)
    b = np.sin(2*a) * 0.4 + 0.5
    li = list((np.array([a]), np.array([b])) for a, b in zip(a, b))
    while True:
        shuffle(li)
        for a, b in li:
            yield a, np.sin(15*a) 

arch = [1] + [30] * 5 + [1]
test = neural_nets.network(arch, bias_vec = True)
trainer = rmsprop_optimizer(test, l_rate = 0.2, l_decay = 0, moment = 0)
# trainer.add_weight_decay(0.0001)
# trainer.add_scaling_noise(0.001)
# trainer.add_gaussian_noise(0.001)
# trainer.add_dropout(0.9)
# trainer.add_maxnorm(17)

for i in range(8):
    trainer(stream(), 5000)
    data = [np.array([a]) for a in np.linspace(-1, 1, 2000)]
    a = np.linspace(-1, 1.0, 2000)
    plt.scatter(a, np.sin(15*a), s = 0.01)
    plt.plot(a, test.predict_all(data))
plt.show()


#"""

