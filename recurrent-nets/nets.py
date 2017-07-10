
import tensorflow as tf
import numpy as np
import numpy.random as rd

def xavier_init(a, b, trainable=True):
    scale = np.sqrt(2/(a + b))
    if trainable:
        return tf.Variable(scale * rd.normal(size=[a,b]), dtype=tf.float32)
    else:
        return tf.Variable(tf.zeros([a, b]), trainable=False)


class lstm_cell:

    def __init__(self, inp, mem, out):
        self.lengths = (inp, mem, out)
        self.s  = tf.Variable(tf.zeros([1, mem]), trainable=False)
        self.l  = tf.Variable(tf.zeros([1, out]), trainable=False)

        self.wa = xavier_init(mem+inp, out)
        self.wb = xavier_init(mem+inp, out)
        self.wc = xavier_init(mem+inp, out)
        self.wd = xavier_init(mem+inp, out)

        self.ba = tf.Variable(tf.zeros([out]))
        self.bb = tf.Variable(tf.zeros([out]))
        self.bc = tf.Variable(tf.zeros([out]))
        self.bd = tf.Variable(tf.zeros([out]))

    def step(self, data):
        true_inp = tf.concat([self.s, data], 1)
        mem_a = tf.matmul(true_inp, self.wa) + self.ba
        mem_b = tf.matmul(true_inp, self.wb) + self.bb
        mem_c = tf.matmul(true_inp, self.wc) + self.bc
        mem_d = tf.matmul(true_inp, self.wd) + self.ba

        self.l *= tf.sigmoid(mem_a)
        self.l += tf.sigmoid(mem_b) * tf.tanh(mem_c)
        return mem_d * tf.tanh(self.l)



class lstm:

    def __init__(self, dims, sess):
        self.sess  = sess
        self.cells = [lstm_cell(a, b, b) for a, b in zip(dims, dims[1:-1])]
        self.outw  = xavier_init(dims[-2], dims[-1])
        self.outb  = tf.Variable(tf.zeros([dims[-1]]))

        self.source = tf.placeholder(tf.float32, [1, 128], name="source")
        self.target = tf.placeholder(tf.float32, [1, 128], name="target")
        self.output = self.run(self.source)
        

        error        = (self.target - self.output)**2
        self.trainer = tf.train.AdamOptimizer(1e-3).minimize(error) 

        self.sess.run(tf.global_variables_initializer())

    def run(self, data):
        curr = data
        for cell in self.cells:

            curr = cell.step(curr)
        curr = tf.matmul(curr, self.outw) + self.outb
        return tf.nn.softmax(curr)


    def clear(self):

        for cell in self.cells:
            inp, out, mem = cell.lengths
            cell.s = tf.zeros(cell.s.shape)
            cell.l = tf.zeros(cell.l.shape)

    def train(self, text):
        #self.clear()
        source_vec = np.zeros([1, 128], dtype=np.float32)
        target_vec = np.zeros([1, 128], dtype=np.float32)
        runerr = 0.1
        for index, (inp, out) in enumerate(zip(text, text[1:])):
            source_vec[0, ord(inp)] = 1.0
            target_vec[0, ord(out)] = 1.0

            correct = tf.equal(tf.argmax(self.target, 1), tf.argmax(self.output, 1))

            
            _, c, err = self.sess.run([self.trainer, tf.argmax(self.output, 1), tf.cast(correct, tf.float32)], feed_dict = {
                self.source: source_vec,
                self.target: target_vec
            })
            runerr += 0.001 * err
            runerr *= 0.999
            
            print(out, chr(c), runerr, flush=True)

            source_vec[0, ord(inp)] = 0.0
            target_vec[0, ord(out)] = 0.0


data = open("test.txt").read()
sess = tf.Session()

dims = [128, 60, 128]
nets = lstm(dims, sess)


nets.train(data)


"""
for i in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):

    print(i)

"""