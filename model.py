from __future__ import print_function
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import seq2seq
from tensorflow.python.ops import rnn_cell


class Model:
    def __init__(self, args):
        self.args = args
        self.dropout = tf.Variable(trainable=False, dtype=tf.float32, initial_value=0)
        cell = rnn_cell.LSTMCell(args.hidden, state_is_tuple=True)
        cell = rnn_cell.MultiRNNCell([cell] * args.num_layers, state_is_tuple=True)
        self.cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=self.dropout)
        self.input_data = tf.placeholder(tf.float32, [args.batch_size, args.seq_length, args.seq_dim])
        self.output_data = tf.placeholder(tf.int32, [args.batch_size])
        self.initial_state = cell.zero_state(args.batch_size, tf.float32)
        with tf.variable_scope('rnn_audio'):
            rnn_weights = tf.get_variable("rnn_weights", [args.hidden, args.num_classes])
            rnn_bias = tf.get_variable("rnn_bias", [args.num_classes])
            with tf.device("/cpu:0"):
                inputs = tf.split(1, args.seq_length, self.input_data)
                inputs = [tf.squeeze(input_, [1]) for input_ in inputs]
        outputs, last_state = seq2seq.rnn_decoder(inputs, self.initial_state, cell, scope='rnn_audio')
        output = outputs[-1]
        self.logits = tf.matmul(output, rnn_weights) + rnn_bias
        self.probabilities = tf.nn.softmax(self.logits)
        loss = seq2seq.sequence_loss_by_example([self.logits],
                                                [self.output_data],
                                                [tf.ones([args.batch_size])],
                                                args.num_classes)
        self.cost = tf.reduce_mean(loss)
        self.final_state = last_state
        self.lr = tf.Variable(0.0, trainable=False)
        train_vars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, train_vars), 5)
        optimizer = tf.train.AdamOptimizer(self.lr)
        self.train_op = optimizer.apply_gradients(zip(grads, train_vars))

    def predict(self, sess, input_data):
        num_batches = (len(input_data) + self.args.batch_size - 1) // self.args.batch_size
        resize = num_batches * self.args.batch_size
        resize_input = np.resize(input_data, (resize, self.args.seq_length, self.args.seq_dim))
        predictions = np.zeros((0, self.args.num_classes))
        for i in range(0, len(resize_input), self.args.batch_size):
            predictions = np.append(predictions,
                                    sess.run(self.probabilities,
                                             feed_dict={self.input_data: resize_input[i:i + self.args.batch_size],
                                                        self.dropout: 1}),
                                    axis=0)
        return predictions[:len(input_data)]

    def train_epoch(self, sess, input_data, output_data, dropout, e):
        num_batches = (len(input_data) + self.args.batch_size - 1) // self.args.batch_size
        resize = num_batches * self.args.batch_size
        resize_input = np.resize(input_data, (resize, self.args.seq_length, self.args.seq_dim))
        resize_output = np.resize(output_data, resize)
        for i in range(0, len(resize_input), self.args.batch_size):
            _, cost = sess.run([self.train_op, self.cost],
                               feed_dict={self.input_data: resize_input[i:i + self.args.batch_size],
                                          self.output_data: resize_output[i:i + self.args.batch_size],
                                          self.lr: self.args.lr * (self.args.decay ** e),
                                          self.dropout: dropout})
            print('batch percentage = %f, batch cost = %f' % (i * 100 / len(resize_input), cost))
