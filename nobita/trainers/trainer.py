"""
    note: check input data type
"""
import importlib
import os

class trainer:

    def __init__(self, net, flow, rate, freq = 1000):
        x, y = next(flow)
        size, _ = x.shape
        same, _ = y.shape
        assert size == same
        self.net = net
        self.rate = rate
        self.flow = flow
        self.size = size
        self.freq = freq

        self.decay = 0.0
        self.dropout = 0.0
        self.maxnorm = 0.0
        self.l_decay = 0.0

    def set_decay(self, decay):
        self.decay = decay

    def set_dropout(self, vec):
        if isinstance(vec, (float, int)):
            vec = self.net.n * [vec]
        self.dropout = vec

    def set_maxnorm(self, cap):
        self.maxnorm = cap


    def header(self):
        start = ["def run(net, iterations, flow):"]
        push, n, bs = start.append, self.net.n, self.net.bs
        push("from numpy import dot")
        push("from numpy.random import uniform as unif")
        push("")
        for i in range(n):
            push(f"a{i}, p{i} = net.funs[{i}]")
            push(f"w{i} = net.ws[{i}]")
            if bs[i] is not None:
                push(f"b{i} = net.bs[{i}]")
            push("")
        push("run_err = 0.0")
        push(f"scale = {self.rate/self.size}")
        return start

    def regular(self):
        norm = []
        push, n, s = norm.append, self.net.n, self.size
        if self.dropout :
            arch, dropout = self.net.arch, self.dropout
            for i in range(n):
                if self.dropout[i]:
                    d = dropout[i]
                    push(f"drop{i} = (unif(size={(s, arch[i])})>{d})/{1 - d}")

        if self.decay:
            push("")
            for i in range(n): push(f"w{i} *= 1 - {self.decay}*scale")

        if self.l_decay: push(f"scale *= {1 - self.l_decay}")

        if self.maxnorm:
            mn = self.maxnorm
            push("")
            m = self.maxnorm
            for i in range(n):
                push(f"norm{i} = (w{i} ** 2).sum(axis=0)")
            for i in range(n):
                push(f"nind{i} = norm{i} > {mn ** 2}")
            for i in range(n):
                push(f"w{i}[:, nind{i}] *= {mn}/norm{i}[nind{i}]**0.5")

        return norm

    def fore_pass(self):
        fore = [""]
        push, n, bs = fore.append, self.net.n, self.net.bs
        for i in range(n):
            if self.dropout and self.dropout[i]:
                push(f"x{i} *= drop{i}")
            if bs[i] is None:
                push(f"x{i+1} = a{i}(dot(x{i}, w{i}))")
            else:
                push(f"x{i+1} = a{i}(dot(x{i}, w{i}) + b{i})")
        push("")
        push(f"l{n} = scale * (y - x{n})")
        return fore


    def back_prop(self):
        back = []
        push, n = back.append, self.net.n
        push("")
        for i in reversed(range(1, n)):
            push(f"d{i} = l{i+1} * p{i}(x{i+1})")
            if self.dropout and self.dropout[i]:
                push(f"l{i} = dot(d{i}, w{i}.T) * drop{i}")
            else:
                push(f"l{i} = dot(d{i}, w{i}.T)")

        push("d0 = l1 * p0(x1)")
        return back


    def update(self):
        seqe = [""]
        push, n, bs = seqe.append, self.net.n, self.net.bs
        for i in range(n):
            push(f"w{i} += dot(x{i}.T, d{i})")
        push("")
        for i in range(n):
            if bs[i] is not None:
                push(f"b{i} += d{i}.sum(axis=0)")
        return seqe

    def notify(self):
        note = [""]
        push, n = note.append, self.net.n
        freq, size, rate = self.freq, self.size, self.rate
        if freq:
            push("run_err *= 0.99")
            push(f"run_err += ((l{n}/scale)**2).sum()/{size}")
            push(f"if i%{freq} == 0: print('iteration', i, 'with error', run_err)")
        return note

    def makeloop(self):
        loop = ["for i in range(iterations):"]
        push = loop.append

        push("x0, y = next(flow)")
        loop.extend(self.regular())
        loop.extend(self.fore_pass())
        loop.extend(self.back_prop())
        loop.extend(self.update())
        if self.freq: loop.extend(self.notify())
        return loop

    @property
    def program(self):
        loop_string = "\t\n\t"+"\n\t\t".join(self.makeloop())
        head_string = "\n\t".join(self.header())
        return head_string+loop_string

    def train(self, iterations):
        program = self.program
        f = open("compiled.py", 'w')
        f.write(program)
        f.close()
        import compiled
        importlib.reload(compiled)
        compiled.run(self.net, iterations, self.flow)
