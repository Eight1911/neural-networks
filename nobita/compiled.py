def run(net, iterations, flow):
	from numpy import dot, zeros
	from numpy.random import uniform as unif
	
	a0, p0 = net.funs[0]
	w0 = net.ws[0]
	wm0 = zeros(shape=(784, 500))
	b0 = net.bs[0]
	bm0 = zeros(shape=(500,))
	
	a1, p1 = net.funs[1]
	w1 = net.ws[1]
	wm1 = zeros(shape=(500, 300))
	b1 = net.bs[1]
	bm1 = zeros(shape=(300,))
	
	a2, p2 = net.funs[2]
	w2 = net.ws[2]
	wm2 = zeros(shape=(300, 10))
	b2 = net.bs[2]
	bm2 = zeros(shape=(10,))
	
	run_err = 0.0
	scale = 0.0001	
	for i in range(iterations):
		x0, y = next(flow)
		
		w0 *= 1 - 1e-05*scale
		w1 *= 1 - 1e-05*scale
		w2 *= 1 - 1e-05*scale
		
		x1 = a0(dot(x0, w0) + b0)
		x2 = a1(dot(x1, w1) + b1)
		x3 = a2(dot(x2, w2) + b2)
		
		l3 = scale * (y - x3)
		
		d2 = l3 * p2(x3)
		l2 = dot(d2, w2.T)
		d1 = l2 * p1(x2)
		l1 = dot(d1, w1.T)
		d0 = l1 * p0(x1)
		
		wm0 *= 0.9
		d0 = 0.0009999999999999998 * dot(x0.T, d0)
		wm0 += d0
		w0 += d0 + 0.9*(wm0)
		wm1 *= 0.9
		d1 = 0.0009999999999999998 * dot(x1.T, d1)
		wm1 += d1
		w1 += d1 + 0.9*(wm1)
		wm2 *= 0.9
		d2 = 0.0009999999999999998 * dot(x2.T, d2)
		wm2 += d2
		w2 += d2 + 0.9*(wm2)
		
		wm0 *= 0.9
		wm0 += 0.0009999999999999998 * d0.sum(axis=0)
		b0 += bm0
		wm1 *= 0.9
		wm1 += 0.0009999999999999998 * d1.sum(axis=0)
		b1 += bm1
		wm2 *= 0.9
		wm2 += 0.0009999999999999998 * d2.sum(axis=0)
		b2 += bm2
		
		run_err *= 0.99
		run_err += ((l3/scale)**2).sum()/100
		if i%1000 == 0: print('iteration', i, 'with error', run_err)