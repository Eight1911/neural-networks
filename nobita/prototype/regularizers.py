# simplicity is the ultimate sophistication

from numpy import dot, sqrt, einsum
from numpy.random import rand


def set_dropout(drop_pairs):
    for vec, p in drop_pairs:
        vec[:] = rand(vec.size) <= p

def scale_weights(weight, p_vec):
    for p, weight in zip(p_vec, weight):
        weight /= p

def descale_weights(weight, p_vec):
    for p, weight in zip(p_vec, weight):
        weight *= p

def dropout_unwind(self):
    p_vec = self.p_vec
    drop_vecs = self.drop_vecs
    batch_size = self.batch_size
    drop_pairs = self.drop_pairs
    return drop_pairs, batch_size, set_dropout


def clipnorm(weights, maxnorm_2):
    for weight in weights:
        # update this part, avoid python loops
        # part_sum += einsum('ji,ji->j', weight, weight)
        for column in weight.T:
            norm = dot(column, column)
            if norm > maxnorm_2:
                column *= sqrt(maxnorm_2 / norm)
    
def clipnorm_unwind(self):
    return clipnorm, self.maxnorm ** 2
