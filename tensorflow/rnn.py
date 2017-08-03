#!/usr/bin/env python
import time
import optparse
import numpy as np
import tensorflow as tf

from random import randint

def get_feed_dict(x_data, y_data=None):
    feed_dict = {}

    if y_data is not None:
        feed_dict[y] = y_data

    # for i in range(x_data.shape[0]):
    #     x[i] = x_data[i, :, :]

    feed_dict[x] = x_data

    return feed_dict


# Parameters
optparser = optparse.OptionParser()
optparser.add_option("-n", "--network_type", default='rnn', help="Network type (rnn, lstm, basic_lstm)")
optparser.add_option("-l", "--hidden_size", default=100, type='int', help="Hidden layer size")
optparser.add_option("-s", "--seq_length", default=30, type='int', help="Sequence length")
optparser.add_option("-b", "--batch_size", default=20, type='int', help="Batch size")
opts = optparser.parse_args()[0]

network_type = opts.network_type
print("Lstm type is: {}".format(network_type))
hidden_size = opts.hidden_size
hidden_size = opts.hidden_size
seq_length = opts.seq_length
batch_size = opts.batch_size

n_batch = 1000
n_samples = batch_size * n_batch 

n_pre_alllocate = 100

# Data
pre_allocate_xinput = np.random.rand(n_pre_alllocate, seq_length, batch_size, hidden_size).astype(np.float32)
ytarget = np.random.rand(batch_size, hidden_size).astype(np.float32)

with tf.device('/gpu:0'):

   #x = [tf.placeholder(tf.float32, [batch_size, hidden_size], name="x") for i in range(seq_length)]
   x = tf.placeholder(tf.float32, [seq_length, batch_size, hidden_size])
   y = tf.placeholder(tf.float32, [batch_size, hidden_size], name="y")

   if network_type == 'rnn':
       cell = tf.nn.rnn_cell.BasicRNNCell(hidden_size)
       output, _cell_state = rnn.dynamic_rnn(cell, x, time_major=True, dtype=tf.float32)
   elif network_type == 'lstm':
       cell = tf.nn.rnn_cell.LSTMCell(hidden_size, hidden_size)
       output, _cell_state = rnn.dynamic_rnn(cell, x, time_major=True, dtype=tf.float32)
   elif network_type == 'basic_lstm':
       cell = tf.contrib.rnn.BasicLSTMCell(hidden_size)
       output, _cell_state = rnn.dynamic_rnn(cell, x, time_major=True, dtype=tf.float32)
   elif network_type == 'cudnn_lstm':
       cell = tf.contrib.cudnn_rnn.CudnnLSTM(
           num_layers=1,
           num_units=hidden_size,
           input_size=hidden_size)
       params_size_t = cell.params_size()
       params = tf.Variable(tf.ones([params_size_t]), validate_shape=False)
       output, _hidden_state, _cell_state = cell(
           input_data=x,
           input_h=tf.zeros([1,batch_size,hidden_size]),
           input_c=tf.zeros([1,batch_size,hidden_size]),
           params=params,
           is_training=True)
   elif network_type == 'fused_lstm':
       cell = tf.contrib.rnn.LSTMBlockFusedCell(hidden_size)
       output, _cell_state = cell(x, dtype=tf.float32)
   else:
       raise Exception('Unknown network! '+network_type)

   print("Compiling...")
   start = time.time()
   cost = tf.reduce_sum((output[-1] - y) ** 2)

   optim = tf.train.GradientDescentOptimizer(0.01)
   train_op = optim.minimize(cost)

   session = tf.Session()
   session.run(tf.initialize_all_variables())
   session.run(train_op, feed_dict=get_feed_dict(pre_allocate_xinput[0], ytarget))
   print("Setup : compile + forward/backward x 1")
   print("--- %s seconds" % (time.time() - start))

   start = time.time()
   for i in range(0, n_batch):
       xinput = pre_allocate_xinput[randint(0, n_pre_alllocate - 1)]
       session.run(output, feed_dict=get_feed_dict(xinput))
   end = time.time()
   print("Forward:")
   print("--- %i samples in %s seconds (%f samples/s, %.7f s/sample) ---" % (n_samples, end - start, n_samples / (end - start), (end - start) / n_samples))

   start = time.time()
   for i in range(0, n_batch):
       xinput = pre_allocate_xinput[randint(0, n_pre_alllocate - 1)]
       session.run(train_op, feed_dict=get_feed_dict(xinput, ytarget))
   end = time.time()
   print("Forward + Backward:")
   print("--- %i samples in %s seconds (%f samples/s, %.7f s/sample) ---" % (n_samples, end - start, n_samples / (end - start), (end - start) / n_samples))
