from numpy.random import normal, shuffle
import numpy as np

"""
think of normalizing output
"""

class stream:

    def __init__(self, data, noise = 0):
        self.noise = noise
        self.data = data

    @property
    def noisy_stream(self):
        noise = self.noise
        data = self.data
        while True:
            rand.shuffle(data)
            for vec, tar in data:
                vec = normal(vec, noise, vec.size)
                yield out_vec, out_tar

    @property
    def stream(self):
        data = self.data
        while True:
            shuffle(data)
            for vec, tar in data:
                yield vec, tar


