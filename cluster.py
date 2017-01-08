from __future__ import print_function
import argparse
import numpy as np
import pickle as pkl
from data import Data
import tensorflow as tf
from model import Model


def one_hot(val, num_classes):
    label = np.zeros(num_classes)
    label[val] = 1
    return label


def accumulate_predict(seg_predict, num_classes, limit):
    accumulated_predict = np.zeros(num_classes)
    for i in range(len(seg_predict)):
        accumulated_predict[seg_predict[i]] += 1
    if limit == -1:
        accumulated_predict = (accumulated_predict > 0) * 1
    else:
        sort_predict = sorted(list(range(num_classes)), key=lambda x: accumulated_predict[x], reverse=True)
        accumulated_predict *= 0
        accumulated_predict[sort_predict[0:int(limit)]] = 1
    return accumulated_predict


def f_score(label, cluster, predict):
    audio_label, audio_predict = cluster_predict(label, cluster, predict)
    precision = np.sum(audio_label * audio_predict, axis=0) / np.sum(audio_predict, axis=0)
    recall = np.sum(audio_label * audio_predict, axis=0) / np.sum(audio_label, axis=0)
    f1 = 2 * precision * recall / (precision + recall)
    print("class wise stats:")
    print(precision)
    print(recall)
    print(f1)
    total_precision = np.sum(audio_label * audio_predict) / np.sum(audio_predict)
    total_recall = np.sum(audio_label * audio_predict) / np.sum(audio_label)
    total_f1 = 2 * total_precision * total_recall / (total_precision + total_recall)
    print("total stats")
    print(total_precision)
    print(total_recall)
    print(total_f1)
    return total_f1


def cluster_predict(label, cluster, predict):
    assert len(label) == len(predict) == np.sum(cluster)
    num_classes = predict.shape[1]
    predict = np.argmax(predict, axis=1)
    audio_label = []
    audio_predict = []
    start = 0
    for no in cluster:
        current_label = label[start]
        if len(current_label.shape) == 0:
            current_label = one_hot(current_label, num_classes)
        num_current_label = np.sum(current_label)
        audio_label.append(current_label)
        current_predict = accumulate_predict(predict[start:start + no], num_classes, num_current_label)
        audio_predict.append(current_predict)
        start += no
    audio_label = np.array(audio_label)
    audio_predict = np.array(audio_predict)
    return [audio_label, audio_predict]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_set_load', type=str, nargs=3, help='path of data and label and cluster in each sample',
                        required=True)
    parser.add_argument('--save_path', type=str, nargs=2, help='path of label and predict clustered', required=True)
    parser.add_argument('--hidden', type=int, help='hidden dimension of rnn typically 256', required=True)
    parser.add_argument('--num_classes', type=int, help='number of classes', required=True)
    parser.add_argument('--batch_size', type=int, help='batch size typically 128 or 64', required=True)
    parser.add_argument('--seq_length', type=int, help='sequence length', required=True)
    parser.add_argument('--seq_dim', type=int, help='sequence dimension', required=True)
    parser.add_argument('--num_layers', type=int, help='number of layers in rnn typically 2', required=True)
    parser.add_argument('--lr', type=float, help='initial learning rate typically .002', required=True)
    parser.add_argument('--decay', type=float, help='decay rate typically .97', required=True)
    parser.add_argument('--restore', type=str, help='path of saved model', required=True)
    args = parser.parse_args()
    model = Model(args)
    data = Data('', 22050)
    data.load(args.data_set_load[0], args.data_set_load[1], args.data_set_load[2])
    sess = tf.Session()
    saver = tf.train.Saver()
    saver.restore(sess, args.restore)
    print('model loaded')
    seg_predict = model.predict(sess, data.input_data)
    clustered_label, clustered_predict = cluster_predict(data.output_data, data.cluster, seg_predict)
    pkl.dump(clustered_label, open(args.save_path[0], 'wb'))
    pkl.dump(clustered_predict, open(args.save_path[1], 'wb'))


if __name__ == '__main__':
    main()
