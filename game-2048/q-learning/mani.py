
from numpy.random import choice
from numpy import zeros, uint8, array, e, empty
import numpy as np

from util import addtile
import util


RANGE16 = np.arange(16)
RANGE4 = np.arange(4)
ABLES = [util.uable, util.dable, util.lable, util.rable]
MOVES = [util.u, util.d, util.l, util.r]

def tovec(arr):
    square = zeros((16, 16), uint8)
    square[RANGE16, arr.reshape(16)] = 1
    square.resize((1, 256))
    return square

def reflect(arr, ind):
    x, y = divmod(ind, 2)
    square = empty((8, 256), uint8)
    target = empty(8)

    lr = np.fliplr(arr) # y = (y + 1) % 2
    ud = np.flipud(arr) # x = (x + 1) % 2
    lu = np.fliplr(ud)  # both lol

    arr = arr.T



import numpy as np


def toarr(square):
    board = zeros(16, uint8)
    square.resize((16, 16))
    board[:] = np.argmax(square, axis = 1)
    square.resize((1, 256))
    return board.reshape((4, 4))


class trainer:

    def __init__(self, net):
        self.net = net


    def play(self):
        "play 2048 using probabilistic recommendations of net"
        states = []
        runn = self.net.__call__
        board = zeros((4, 4), uint8)
        while True:
            addtile(board)

            able = array([f(board) for f in ABLES])
            if not able.sum(): return states
            vec = tovec(board)
            pvec = runn(vec).reshape(4) * able
            ind = choice(RANGE4, p=pvec / pvec.sum())
            states.append((vec, pvec, ind))
            MOVES[ind](board)


    def stream(self):
        run = self.play
        running = len(run())
        target = zeros(4)
        for i in range(100000000000):
            if i % 1 == 0: print(running)
            game = run()
            running *= 0.995
            running += 0.005 * len(game)
            reinforce = e**((len(game) - running) / 1000)
            for vec, pvec, ind in game:
                pvec[ind] *= reinforce
                yield vec.reshape((1,256)), (pvec/pvec.sum()).reshape((1,4))


k = arange(16).reshape((4, 4))
k = np.flipud(np.fliplr(k))
print(k)
