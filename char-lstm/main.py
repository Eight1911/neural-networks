
import tensorflow as tf
import numpy as np
import random

from stream import sample

session = tf.Session()

f32  = tf.float32

class lstm:

    def __init__(self, arch, time, batch):
        self.arch = arch
        self.inte = arch[1: -1]
        self.time = time
        self.batch = batch

        self.inp_size = arch[ 0]
        self.out_size = arch[-1]

        self.var_dict = self.init_vars()

    def init_vars(self):

        def run_through(weights, vec, state):
            inp_w, inp_b, out_w, out_b, stacked = weights

            output = tf.matmul(vec, inp_w) + inp_b
            output = tf.nn.elu(output)
            output, state = stacked(output, state=state)
            output = tf.matmul(output, out_w) + out_b

            return output, state

        def init(self):
            x = tf.placeholder(f32, (self.time  , self.batch, self.inp_size))
            
            base_inp_w = np.random.randn(*self.arch[ :2]) / sum(self.arch[ :2])
            base_out_w = np.random.randn(*self.arch[-2:]) / sum(self.arch[-2:])

            inp_w, inp_b = tf.Variable(
                base_inp_w.astype('float32')
            ), tf.Variable(np.zeros(self.arch[ 1], dtype='float32'))

            out_w, out_b = tf.Variable(
                base_out_w.astype('float32')
            ), tf.Variable(np.zeros(self.arch[-1], dtype='float32'))

            stacked = tf.nn.rnn_cell.MultiRNNCell([
                tf.nn.rnn_cell.LSTMCell(n) for n in self.inte 
            ])
            states = [None] * (self.time + 1)
            output = [None] * (self.time + 1)
            states[0] = stacked.zero_state(self.batch, tf.float32)

            weights = (inp_w, inp_b, out_w, out_b, stacked)
            print("running through")
            for i in range(self.time):
                if i % 25 == 0: print()
                print(".", end="", flush=True)
                output[i+1], states[i+1] = run_through(weights, x[i], states[i])

            print()
            print("end run through")
            return {
                'x'     : x,
                'output': output,
                'states': states,
                'weight': weights
            }

        def iter_vec(v):
            return [v[i] for i in range(v.shape[0])]

        def loss(output, y):
            scale    = len(output) - 1
            priority = np.linspace(0, 1, scale) ** 0.5
            priority *= 10/sum(priority)
            print("starting loss")
            l        = tf.nn.softmax_cross_entropy_with_logits
            loss_m   = lambda y, out, p: p * l(labels=y, logits=out)
            loss_e   = lambda e : loss_m(*e)

            iterator = [*zip(iter_vec(y), output[1:], priority)]

            loss     = sum(map(loss_e, iterator)) / scale
            print("starting optimizer")
            optimizer = tf.train.AdamOptimizer(3e-5)
            print("started optimizer")
            print("calculating gradients... (this may take a couple minutes)")
            optimizer = optimizer.minimize(loss)
            print("gradients calculated")

            return loss, optimizer

        def main(self):
            init_dict = init(self)
            error     = loss(init_dict['output'], init_dict['x'][1:])
            init_dict['loss'], init_dict['optimizer'] = error
            return init_dict

        return main(self)


    def run(self, data_seq):
        print("initializing variables")
        session.run(tf.global_variables_initializer())
        print("initialized variables")

        hide_tuple = (self.var_dict['loss'],
                      self.var_dict['optimizer'])

        show_tuple = (self.var_dict['loss'],
                      self.var_dict['output'][1:],
                      self.var_dict['optimizer'])

        run_error = 0
        for i in range(0, 1000000):
            data = data_seq(self.batch)

            vdict  = { 
                self.var_dict['x'] : data, 
            }

            loss, output, _ = session.run(show_tuple, vdict)
            run_error += 0.01 * loss.sum()
            run_error *= 0.99
            print(i, run_error)

            for v in output:
                print(chr(np.argmax(v[0])), end = "")
            print()
        


time = 100
batch = 3
print("check point one")
net = lstm([128, 100, 100, 100, 128], time, batch)
print("check point two")
net.run(sample(time))
