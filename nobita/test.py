from trainers.momentum import nesterov as trainer
from networks.network import network


from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

import numpy as np
def stream():
    while True:
        yield mnist.train.next_batch(100)



flow = stream()
arch = [784, 500, 300, 10]
bias = True
acts = None

net = network(arch)
train = trainer(net, flow, 0.01, 0.9, 1000)
#train.l_decay = 0.0001
#train.set_dropout([0.5, 0, 0, 0])
train.set_decay(1e-5)
#train.set_maxnorm(5)

train.train(100000)
result = net(mnist.test.images)
preds = np.argmax(result, axis = 1)
label = np.argmax(mnist.test.labels, axis = 1)
print(label.size)
print("results:", np.equal(preds, label).sum())
"""
train.set_dropout([0.1, 0.1, 0.1, 0.1])
for i in range(6):
    train.train(5000)
    result = net(mnist.test.images)
    preds = np.argmax(result, axis = 1)
    label = np.argmax(mnist.test.labels, axis = 1)
    print("results:", np.equal(preds, label).sum())
"""
