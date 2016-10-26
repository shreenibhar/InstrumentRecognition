from __future__ import print_function

import argparse

import tensorflow as tf

from data import Data
from model import Model

parser = argparse.ArgumentParser()
parser.add_argument('--hidden', type=int, default=256, help='hidden dimension of rnn')
parser.add_argument('--num_classes', type=int, help='number of classes', required=True)
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--seq_length', type=int, help='sequence length', required=True)
parser.add_argument('--seq_dim', type=int, help='sequence dimension', required=True)
parser.add_argument('--num_layers', type=int, default=2, help='number of layers in rnn')
parser.add_argument('--epoch', type=int, default=50, help='number of epoch')
parser.add_argument('--lr', type=float, default=0.002, help='initial learning rate')
parser.add_argument('--decay', type=float, default=0.97, help='decay rate')
parser.add_argument('--data_set_load', type=str, nargs=2, help='path of output and label', required=True)
parser.add_argument('--restore', type=str, default=None, help='path of save')
args = parser.parse_args()
model = Model(args)
data = Data('', 22050)
data.load(args.data_set_load[0], args.data_set_load[1])
sess = tf.Session()
saver = tf.train.Saver()
if args.restore is not None:
    saver.restore(sess, args.restore)
    print('model loaded')
else:
    sess.run(tf.initialize_all_variables())
    print('model initialized')
model.train_epochs(sess, data.input_data, data.output_data, 500)
