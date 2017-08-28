import random
import numpy as np

def one_hot(n):
    return np.eye(n)


def sample(time):

    hot_o = [i for i in np.eye(128)]
    hot_f = lambda c : hot_o[ord(c)]

    file = open('sample/tale.txt', 'r').read()
    text = [*map(hot_f, file[:479801])]
    pads = [hot_f(" ")] * time
    text = pads + text + pads
    text = np.array(text)
    size = len(text) - time

    def sampler(batch):

        inds = [random.randrange(size) for _ in range(batch)]
        base = [text[ind: ind + time] for ind in inds]
        return np.array(base).swapaxes(0, 1)

    return sampler

