import random
import numpy as np
import numpy.random as rd
import tensorflow as tf
import sklearn.manifold as manifold


def sammon(dists, dim=2, iterations=5000, lrate=0.015, start='mds'):
    n = len(dists)
    if type(start) == str and start == 'mds':
        fitter = manifold.MDS(
            dim, 
            dissimilarity="precomputed", 
            verbose=1, 
            max_iter=1500, 
            eps=1e-5)
        start = fitter.fit_transform(dists)
    elif type(start) == str and start == 'random':
        start = rd.randn(n, dim)

    with tf.scope('sammon-mapper'):
        tf.reset_default_graph()
        total_sum = dists.sum()
        cdists = tf.constant(dists) + 1e-8
        cdists2 = cdists * cdists
        points = tf.Variable(start)
        points2 = tf.reduce_sum(points * points, 1)
        points2 = tf.reshape(points2, [-1, 1])
        distmat = points2 - 2*tf.matmul(points, tf.transpose(points)) + tf.transpose(points2)
        distmat = tf.maximum(distmat, 0.0) ** 0.5
        loss = tf.reduce_sum((cdists - distmat)**2 / cdists)
        trainer = tf.train.AdagradOptimizer(lrate)
        minimizer = trainer.minimize(loss)
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

    for i in range(iterations):
        _, l = sess.run([minimizer, loss])
        if i % 100 == 0:
            print(i, l / total_sum)
    print("final loss", l/ total_sum)
    return sess.run(points)