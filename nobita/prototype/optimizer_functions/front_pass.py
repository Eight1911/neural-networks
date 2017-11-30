from numpy.random import normal
from numpy import dot, outer



def front_pass(vecs, front_arch, d_rate):
	vec = vecs[0]
	for index, act, weight, bias in front_arch:
		vec = ( act(dot(weight, vec)) 
				if bias is None
				else act(dot(weight, vec) + bias) )
		vecs[index] = vec

def decay_pass(vecs, front_arch, d_rate):
	vec = vecs[0]
	for index, act, weight, bias in front_arch:
		weight *= d_rate
		vec = ( act(dot(weight, vec)) 
				if bias is None
				else act(dot(weight, vec) + bias) )
		vecs[index] = vec

def dropout_front_pass(vecs, front_arch, d_rate):
	vec = vecs[0]
	for index, act, weight, bias, drop_vec in front_arch:
		vec *= drop_vec
		vec = ( act(dot(weight, vec)) 
				if bias is None
				else act(dot(weight, vec) + bias) )
		vecs[index] = vec

def dropout_decay_pass(vecs, front_arch, d_rate):
	vec = vecs[0]
	for index, act, weight, bias, drop_vec in front_arch:
		vec *= drop_vec
		weight[:, drop_vec == 1] *= d_rate
		vec = ( act(dot(weight, vec)) 
				if bias is None 
				else act(dot(weight, vec) + bias) )
		vecs[index] = vec




def gauss_front_pass(vecs, front_arch, d_rate):
	vec = vecs[0]
	for index, act, weight, bias, gauss_sigma in front_arch:
		vec += normal(0, gauss_sigma, vec.size)
		vec = ( act(dot(weight, vec))
				if bias is None
				else act(dot(weight, vec) + bias) )
		vecs[index] = vec

def gauss_decay_pass(vecs, front_arch, d_rate):
	vec = vecs[0]
	for index, act, weight, bias, gauss_sigma in front_arch:
		vec += normal(0, gauss_sigma, vec.size)
		weight *= d_rate
		vec = ( act(dot(weight, vec)) 
				if bias is None
				else act(dot(weight, vec) + bias) )
		vecs[index] = vec

def gauss_dropout_front_pass(vecs, front_arch, d_rate):
	vec = vecs[0]
	for index, act, weight, bias, drop_vec, gauss_sigma in front_arch:
		vec += normal(0, gauss_sigma, vec.size)
		vec *= drop_vec
		vec = ( act(dot(weight, vec)) 
				if bias is None
				else act(dot(weight, vec) + bias) )
		vecs[index] = vec

def gauss_dropout_decay_pass(vecs, front_arch, d_rate):
	vec = vecs[0]
	for index, act, weight, bias, drop_vec, gauss_sigma in front_arch:
		vec += normal(0, gauss_sigma, vec.size)
		vec *= drop_vec
		weight[:, drop_vec == 1] *= d_rate
		vec = ( act(dot(weight, vec)) 
				if bias is None 
				else act(dot(weight, vec) + bias) )
		vecs[index] = vec





def scaling_front_pass(vecs, front_arch, d_rate):
	vec = vecs[0]
	for index, act, weight, bias, scale_sigma, scale_vec in front_arch:
		scale_vec[:] = normal(1, scale_sigma, scale_vec.size)
		vec *= scale_vec
		vec = ( act(dot(weight, vec))
				if bias is None
				else act(dot(weight, vec) + bias) )
		vecs[index] = vec

def scaling_decay_pass(vecs, front_arch, d_rate):
	vec = vecs[0]
	for index, act, weight, bias, scale_sigma, scale_vec in front_arch:
		scale_vec[:] = normal(1, scale_sigma, scale_vec.size)
		vec *= scale_vec
		weight *= d_rate
		vec = ( act(dot(weight, vec)) 
				if bias is None
				else act(dot(weight, vec) + bias) )
		vecs[index] = vec

def scaling_dropout_front_pass(vecs, front_arch, d_rate):
	vec = vecs[0]
	for index, act, weight, bias, drop_vec, scale_sigma, scale_vec in front_arch:
		scale_vec[:] = normal(1, scale_sigma, scale_vec.size)
		vec *= scale_vec * drop_vec
		vec = ( act(dot(weight, vec)) 
				if bias is None
				else act(dot(weight, vec) + bias) )
		vecs[index] = vec

def scaling_dropout_decay_pass(vecs, front_arch, d_rate):
	vec = vecs[0]
	for index, act, weight, bias, drop_vec, scale_sigma, scale_vec in front_arch:
		scale_vec[:] = normal(1, scale_sigma, scale_vec.size)
		vec *= scale_vec * drop_vec
		weight[:, drop_vec == 1] *= d_rate
		vec = ( act(dot(weight, vec)) 
				if bias is None 
				else act(dot(weight, vec) + bias) )
		vecs[index] = vec




def scaling_gauss_front_pass(vecs, front_arch, d_rate):
	vec = vecs[0]
	for index, act, weight, bias, gauss_sigma, scale_sigma, scale_vec in front_arch:
		scale_vec[:] = normal(1, scale_sigma, scale_vec.size)
		vec += normal(0, gauss_sigma, vec.size)
		vec *= scale_vec
		vec = ( act(dot(weight, vec)) 
				if bias is None
				else act(dot(weight, vec) + bias) )
		vecs[index] = vec

def scaling_gauss_decay_pass(vecs, front_arch, d_rate):
	vec = vecs[0]
	for index, act, weight, bias, gauss_sigma, scale_sigma, scale_vec in front_arch:
		scale_vec[:] = normal(1, scale_sigma, scale_vec.size)
		vec += normal(0, gauss_sigma, vec.size)
		vec *= scale_vec
		weight *= d_rate
		vec = ( act(dot(weight, vec)) 
				if bias is None
				else act(dot(weight, vec) + bias) )
		vecs[index] = vec

def scaling_gauss_dropout_front_pass(vecs, front_arch, d_rate):
	vec = vecs[0]
	for index, act, weight, bias, drop_vec, gauss_sigma, scale_sigma, scale_vec in front_arch:
		scale_vec[:] = normal(1, scale_sigma, scale_vec.size)
		vec += normal(0, gauss_sigma, vec.size)
		vec *= scale_vec * drop_vec
		vec = ( act(dot(weight, vec)) 
				if bias is None
				else act(dot(weight, vec) + bias) )
		vecs[index] = vec

def scaling_gauss_dropout_decay_pass(vecs, front_arch, d_rate):
	vec = vecs[0]
	for index, act, weight, bias, drop_vec, gauss_sigma, scale_sigma, scale_vec in front_arch:
		scale_vec[:] = normal(1, scale_sigma, scale_vec.size)
		vec += normal(0, gauss_sigma, vec.size)
		vec *= scale_vec * drop_vec
		weight[:, drop_vec == 1] *= d_rate
		vec = ( act(dot(weight, vec)) 
				if bias is None 
				else act(dot(weight, vec) + bias) )
		vecs[index] = vec







