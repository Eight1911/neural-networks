import random
import numpy as np
import util

RANGE16 = np.arange(16)
RANGE4 = np.arange(4)
MOVES = [util.u, util.d, util.l , util.r]
ABLES = [util.uable, util.dable, util.lable , util.rable]


def tovec(board):
    vec = np.zeros((16, 16), np.uint8)
    vec[RANGE16, board.reshape(16)] = 1
    return vec.reshape((1, 256))

def tflip(arr):
    ret = np.zeros((8, 16, 16), np.uint8)
    temp = arr
    ret[0, RANGE16, temp.reshape(16)] = 1
    temp = np.flipud(arr)
    ret[1, RANGE16, temp.reshape(16)] = 1
    temp = np.fliplr(temp)
    ret[2, RANGE16, temp.reshape(16)] = 1
    temp = np.fliplr(arr)
    ret[3, RANGE16, temp.reshape(16)] = 1

    arr = arr.T
    temp = arr
    ret[4, RANGE16, temp.reshape(16)] = 1
    temp = np.flipud(arr)
    ret[5, RANGE16, temp.reshape(16)] = 1
    temp = np.fliplr(temp)
    ret[6, RANGE16, temp.reshape(16)] = 1
    temp = np.fliplr(arr)
    ret[7, RANGE16, temp.reshape(16)] = 1
    return ret.reshape((8, 256))



class gamer:


    def __init__(self, net):
        self.net = net


    def best(self, board):

        def score(board, n):
            index = np.where(board == 0)
            count = index[0].size
            if not count: return -70.0
            score2, score4 = 0.0, 0.0
            for i, j in zip(*index):
                board[i,j] = 1
                score2 += move(board, n - 1)
                board[i,j] = 2
                score4 += move(board, n - 1)
            return (0.9*score2 + 0.1*score4) / count

        def move(board, n):
            runn = self.net.__call__
            ables = [f(board) for f in ABLES]
            blanc = [-50.0] * 4
            for i, f in enumerate(MOVES):
                if ables[i]:
                    temp = board.copy()
                    f(temp)
                    blanc[i] = score(temp, n) if n else runn(tovec(temp))[0, 0]
            return max(blanc)

        def main(board):
            ables = [f(board) for f in ABLES]
            s = sum(ables)
            if not s:
                return None
            elif s == 1:
                return ables.index(True)
            blanc = np.repeat(-100.0, 4)
            for i, f in enumerate(MOVES):
                if ables[i]:
                    temp = board.copy()
                    f(temp)
                    blanc[i] = score(temp, 2)
            ind = np.argmax(blanc)
            return ind

        return main(board)


    def play(self, start=None):
        states = []
        runn = self.net.__call__
        board = np.zeros((4, 4), np.uint8) if start is None else start
        ind = True
        while True:
            print(board)
            util.addtile(board)
            ind = self.best(board)
            states.append(board.copy())
            if ind is None:
                return states
            MOVES[ind](board)


    def rootedplay(self, start=None):
        states = []
        runn = self.net.__call__
        board = np.zeros((4, 4), np.uint8) if start is None else start
        count = 0
        while True:
            util.addtile(board)
            for i, f in enumerate(ABLES):
                if f(board):
                    temp = board.copy()
                    MOVES[i](temp)
                    yield count, self.play(temp)
            print(count)
            print(board)
            print(self.net(tovec(board)))
            ind = self.best(board)
            states.append(board.copy())
            if ind is None:
                print("Full:", len(states))
                print(board)
                yield 0, states
                break
            MOVES[ind](board)
            count += 1


    def stream(self):
        ret = np.empty((8, 1))
        running = 0
        while True:
            for count, game in self.rootedplay():
                n = len(game)

                running *= 0.99
                running += 0.01*(n + count)
                print(n + count, running)

                for i, board in enumerate(game):
                    ret[:] = np.log((n - i)) - 5
                    yield tflip(board), ret
