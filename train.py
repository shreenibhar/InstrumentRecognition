from __future__ import print_function
import argparse
from data import Data
import tensorflow as tf
from model import Model
from cluster import f_score

parser = argparse.ArgumentParser()
parser.add_argument('--hidden', type=int, help='hidden dimension of rnn typically 256', required=True)
parser.add_argument('--num_classes', type=int, help='number of classes including the null class', required=True)
parser.add_argument('--batch_size', type=int, help='batch size typically 128 or 64', required=True)
parser.add_argument('--seq_length', type=int, help='sequence length', required=True)
parser.add_argument('--seq_dim', type=int, help='sequence dimension', required=True)
parser.add_argument('--num_layers', type=int, help='number of layers in rnn typically 2', required=True)
parser.add_argument('--epoch', type=int, help='number of epoch', required=True)
parser.add_argument('--lr', type=float, help='initial learning rate typically .002 0r .003', required=True)
parser.add_argument('--decay', type=float, help='decay rate typically .97', required=True)
parser.add_argument('--data_set_load', type=str, nargs=3, help='path of output and label and cluster in each sample',
                    required=True)
parser.add_argument('--restore', type=str, default=None, help='path of save')
args = parser.parse_args()
model = Model(args)
train_data = Data('', 22050)
train_data.load(args.data_set_load[0], args.data_set_load[1], args.data_set_load[2])
test_data = train_data.divide_cluster(75)
train_data.shuffle()
sess = tf.Session()
saver = tf.train.Saver()
if args.restore is not None:
    saver.restore(sess, args.restore)
    print('model loaded')
else:
    sess.run(tf.initialize_all_variables())
    print('model initialized')
max_score = -1
for e in range(args.epoch):
    print("Epoch %d:" % e)
    model.train_epoch(sess, train_data.input_data, train_data.output_data, 0.5, e)
    predict = model.predict(sess, test_data.input_data)
    score = f_score(test_data.output_data, test_data.cluster, predict)
    if score > max_score:
        saver.save(sess, 'max_model.ckpt')
        max_score = score
        print('max model saved')
