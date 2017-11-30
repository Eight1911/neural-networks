from numpy.random import normal
from numpy import dot, outer


def back_prop(deltas, vecs, error, back_arch):
	# by default, back_arch is stored
	# in the reverse position
	for index, prime, weight in back_arch:
		delta = error * prime(vecs[index])
		deltas[index - 1] = delta
		error = dot(delta, weight)


def dropout_back_prop(deltas, vecs, error, back_arch):
	for index, prime, weight, drop_vec in back_arch:
		delta = error * prime(vecs[index])
		deltas[index - 1] = delta
		error = dot(delta, weight) * drop_vec


def scale_back_prop(deltas, vecs, error, back_arch):
	for index, prime, weight, scale_vec in back_arch:
		delta = error * prime(vecs[index])
		deltas[index - 1] = delta
		error = dot(delta, weight) * scale_vec


def scale_dropout_back_prop(deltas, vecs, error, back_arch):
	for index, prime, weight, drop_vec, scale_vec in back_arch:
		delta = error * prime(vecs[index])
		deltas[index - 1] = delta
		error = dot(delta, weight) * drop_vec * scale_vec



