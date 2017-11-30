from trainers.trainer import trainer

class momentum(trainer):

    def __init__(self, net, flow, rate, moment=0.9, freq=1000):
        trainer.__init__(self, net, flow, rate, freq)
        self.moment = moment

    def update(self):
        seqe = [""]
        push, n, bs = seqe.append, self.net.n, self.net.bs
        for i in range(n):
            push(f"wm{i} *= {self.moment}")
            push(f"wm{i} += {self.rate * (1-self.moment)} * dot(x{i}.T, d{i})")
            push(f"w{i} += wm{i}")
        push("")
        for i in range(n):
            if bs[i] is not None:
                push(f"wm{i} *= {self.moment}")
                push(f"wm{i} += {self.rate * (1-self.moment)} * d{i}.sum(axis=0)")
                push(f"b{i} += bm{i}")
        return seqe

    def header(self):
        start = ["def run(net, iterations, flow):"]
        push, n, bs, ws = start.append, self.net.n, self.net.bs, self.net.ws
        push("from numpy import dot, zeros")
        push("from numpy.random import uniform as unif")
        push("")
        for i in range(n):
            push(f"a{i}, p{i} = net.funs[{i}]")
            push(f"w{i} = net.ws[{i}]")
            push(f"wm{i} = zeros(shape={ws[i].shape})")
            if bs[i] is not None:
                push(f"b{i} = net.bs[{i}]")
                push(f"bm{i} = zeros(shape={bs[i].shape})")
            push("")
        push("run_err = 0.0")
        push(f"scale = {self.rate/self.size}")
        return start


class nesterov(trainer):

    def __init__(self, net, flow, rate, moment=0.9, freq=1000):
        trainer.__init__(self, net, flow, rate, freq)
        self.moment = moment

    def update(self):
        seqe = [""]
        push, n, bs = seqe.append, self.net.n, self.net.bs
        for i in range(n):
            push(f"wm{i} *= {self.moment}")
            push(f"d{i} = {self.rate * (1-self.moment)} * dot(x{i}.T, d{i})")
            push(f"wm{i} += d{i}")
            push(f"w{i} += d{i} + {self.moment}*(wm{i})")
        push("")
        for i in range(n):
            if bs[i] is not None:
                push(f"wm{i} *= {self.moment}")
                push(f"wm{i} += {self.rate * (1-self.moment)} * d{i}.sum(axis=0)")
                push(f"b{i} += bm{i}")
        return seqe

    def header(self):
        start = ["def run(net, iterations, flow):"]
        push, n, bs, ws = start.append, self.net.n, self.net.bs, self.net.ws
        push("from numpy import dot, zeros")
        push("from numpy.random import uniform as unif")
        push("")
        for i in range(n):
            push(f"a{i}, p{i} = net.funs[{i}]")
            push(f"w{i} = net.ws[{i}]")
            push(f"wm{i} = zeros(shape={ws[i].shape})")
            if bs[i] is not None:
                push(f"b{i} = net.bs[{i}]")
                push(f"bm{i} = zeros(shape={bs[i].shape})")
            push("")
        push("run_err = 0.0")
        push(f"scale = {self.rate/self.size}")
        return start
