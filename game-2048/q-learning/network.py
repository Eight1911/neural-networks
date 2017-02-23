from numpy.random import normal as randn
import numpy as np
from numpy import dot
import tensorflow as tf


def numelu(vec):
    ret = vec.copy()
    ret[vec < 1] = np.exp(vec[vec < 1]) - 1
    return ret

class nnet:

    def __init__(self, arch, acts, varacts):
        self.n = len(arch) - 1

        self.acts = acts
        self.varacts = varacts
        self.dims = list(zip(arch, arch[1:]))
        self.arch = arch
        self.initialize()
        self.sess = tf.Session()
        self.run = self.sess.run
        self.makemodel()


    def initialize(self):
        # xavier weight initialization
        ws, bs = [0] * self.n, [0] * self.n
        for i, (col, row) in enumerate(self.dims):
            factor = 2 / np.sqrt(row + col)
            ws[i] = factor * randn(size=(col, row))
            bs[i] = np.zeros(row)
        self.ws, self.bs = ws, bs
        self.varws = list(map(tf.Variable, ws))
        self.varbs = list(map(tf.Variable, bs))

    def __call__(self, inp):
        ws, bs, acts = self.ws, self.bs, self.acts
        y = inp
        for a, w, b in zip(acts, ws, bs):
            y = a(dot(y, w) + b)
        return y

    def makemodel(self):
        x = tf.placeholder(tf.float64, [None, self.arch[0]])
        y_ = tf.placeholder(tf.float64, [None, self.arch[-1]])
        ws, bs, acts = self.varws, self.varbs, self.varacts
        y = x
        for a, w, b in zip(acts, ws, bs):
            y = a(tf.matmul(y, w) + b)

        loss = tf.reduce_mean((y - y_)**2)
        trainer = tf.train.AdamOptimizer(1e-3).minimize(loss)
        self.run(tf.global_variables_initializer())
        self.trainset = [x, y_, y, loss, trainer]

    def train(self, flow, iterations):
        running = 0
        x, y_, y, loss, trainer = self.trainset
        ws, bs = self.varws, self.varbs
        for i, (arr, batch_xs, batch_ys) in zip(range(iterations), flow):
            _, l, self.ws[:], self.bs[:]  = self.run((trainer, loss, ws, bs), feed_dict={x: batch_xs, y_: batch_ys})
            running *= 0.999
            running += l
            if i % 10 == 0:
                print("", int(running) / 1000, arr.sum(), flush=True)


import mani


arch = [256, 100, 100, 100, 100, 100, 100, 1]
acts = [numelu] * 30
varacts = [tf.nn.elu] * 30
net = nnet(arch, acts, varacts)

streamer = mani.trainer(net)

flow = streamer.stream()

net.train(flow, 1000000)
