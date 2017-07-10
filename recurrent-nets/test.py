import tensorflow as tf
import numpy as np


#print(state.size)
"""
state = tf.zeros([batch_size, lstm.state_size])
probabilities = []
loss = 0.0

for current_batch_of_words in words_in_dataset:
    # The value of state is updated after processing each batch of words.
    output, state = lstm(current_batch_of_words, state)

    # The LSTM output can be used to make next word predictions
    logits = tf.matmul(output, softmax_w) + softmax_b
    probabilities.append(tf.nn.softmax(logits))
    loss += loss_function(probabilities, target_words)

"""
class lstm_generator:

    def __init__(self, size, depth, edge_size, sess, batch=1, loss=tf.nn.softmax_cross_entropy_with_logits):
        self.sess  = sess
        self.loss  = loss
        self.size  = size
        self.depth = depth

        self.cells = [tf.nn.rnn_cell.LSTMCell(size) for i in range(depth)]
        self.multi = tf.nn.rnn_cell.MultiRNNCell(self.cells)

        self.inp_size, self.out_size = edge_size

        self.state =  self.multi.zero_state(batch, tf.float32)

        scale = np.sqrt(2/size)
        self.w_inp = tf.Variable(np.random.normal(size=[self.inp_size, size])/scale, dtype=tf.float32)
        self.w_out = tf.Variable(np.random.normal(size=[size, self.out_size])/scale, dtype=tf.float32)

        self.b_inp = tf.Variable(tf.zeros(size))
        self.b_out = tf.Variable(tf.zeros(self.out_size))

        self.source = tf.placeholder(tf.float32, [batch, self.inp_size])
        self.target = tf.placeholder(tf.float32, [batch, self.out_size])
        self.output = self.model()

        self.error  = loss(labels=self.target, logits=self.output)
        self.runerr = 1.0
        self.trainer = tf.train.AdadeltaOptimizer(1e-2).minimize(self.error)
        print("LSTM cells initialized")


    def model(self):
        state = self.state
        inp = self.source
        out = tf.matmul(inp, self.w_inp) + self.b_inp
        out = tf.nn.elu(out)
        out, self.state = self.multi(out, self.state)
        out = tf.matmul(out, self.w_out) + self.b_out
        return out

    def train_model(self):
        state = self.state
        inp = self.source
        out = tf.matmul(inp, self.w_inp) + self.b_inp
        out = tf.nn.elu(out)
        out, new_state = self.multi(out, self.state)
        out = tf.matmul(out, self.w_out) + self.b_out
        return out, new_state



    def train(self, text):
        source_vec = np.zeros([1, 128], dtype=np.float32)
        target_vec = np.zeros([1, 128], dtype=np.float32)

        char_output = tf.argmax(self.output, 1)
        correct = tf.equal(char_output, tf.argmax(self.output, 1))
        correct = tf.cast(correct, tf.float32)
        stream = list(enumerate(list(zip(text, text[1:]))))

        while True:
            for index, (inp, out) in stream:
                source_vec[0, ord(inp)] = 1.0
                target_vec[0, ord(out)] = 1.0
                _, c, err = self.sess.run([self.trainer, char_output, self.error], feed_dict = {
                    self.source: source_vec,
                    self.target: target_vec
                })
                print(chr(c), end ="", flush=True)
                self.runerr += 0.01 * err
                self.runerr *= 0.99
                
                if index % 10000 == 0:
                    print(f"\n\n{self.runerr}\n\n", flush=True)

                source_vec[0, ord(inp)] = 0.0
                target_vec[0, ord(out)] = 0.0


data = open("test.txt").read()
print(len(data))
with tf.Session() as sess:
    lstm = lstm_generator(10, 4, (128, 128), sess=sess)
    sess.run(tf.global_variables_initializer())
    while True:
        lstm.train("coffee peanuts coffee peanuts cofffee peanuts")


