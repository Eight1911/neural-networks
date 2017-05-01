from numpy.random import normal as randn
import numpy as np
from numpy import dot
import tensorflow as tf

import pickle


class nnet:

    def __init__(self, arch, acts, varacts, re=False):
        self.n = len(arch) - 1

        self.acts = acts
        self.varacts = varacts
        self.dims = list(zip(arch, arch[1:]))
        self.arch = arch
        if re:
            self.initialize()
            print("initialized")
        else:
            try:
                self.load()
                print("loaded")
            except:
                self.initialize()
                print("initialized")

        self.sess = tf.Session()
        self.run = self.sess.run


    def save(self):
        f = open('weights.p', 'wb')
        data = (self.ws, self.bs)
        pickle.dump(data, f)
        f.close()


    def load(self):
        ws, bs = pickle.load(open('weights.p', 'rb'))
        self.initialize(ws, bs)


    def initialize(self, ws=None, bs=None):
        # xavier weight initialization
        if ws and bs:
            assert len(ws) == len(bs) == len(self.dims)
            for i, (col, row) in enumerate(self.dims):
                assert ws[i].shape == (col, row)
                assert bs[i].shape == (row,)
        else:
            ws, bs = [0] * self.n, [0] * self.n
            for i, (col, row) in enumerate(self.dims):
                factor = 1 / np.sqrt(row + col)
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


    def makemodel(self, l_rate):
        x = tf.placeholder(tf.float64, [None, self.arch[0]])
        y_ = tf.placeholder(tf.float64, [None, self.arch[-1]])
        ws, bs, acts = self.varws, self.varbs, self.varacts
        y = x
        for a, w, b in zip(acts, ws, bs):
            y = a(tf.matmul(y, w) + b)

        loss = tf.reduce_mean((y - y_)**2)
        trainer = tf.train.MomentumOptimizer(l_rate, 0.9999).minimize(loss)
        self.run(tf.global_variables_initializer())
        return [x, y_, y, loss, trainer]


    def train(self, flow, iterations, l_rate):
        running = 0
        x, y_, y, loss, trainer = self.makemodel(l_rate)
        ws, bs = self.varws, self.varbs
        for i, (xs, ys) in zip(range(iterations), flow):
            if i % 100:
                _, l = self.run((trainer, loss), feed_dict={x: xs, y_: ys})
            else:
                if not i % 1000: self.save()
                _, l, self.ws[:], self.bs[:]  = self.run((trainer, loss, ws, bs), feed_dict={x: xs, y_: ys})
            running *= 0.999
            running += l


import mani

arch = [256, 130, 50, 50, 1]
acts = [lambda x: np.maximum(0.1*x, x)] * 3  + [lambda x: x]
varacts = [lambda x: tf.maximum(0.1*x, x)] * 3 + [lambda x: x]
net = nnet(arch, acts, varacts)

streamer = mani.gamer(net)
flow = streamer.stream()


net.train(flow, 10000000, 1e-8)
