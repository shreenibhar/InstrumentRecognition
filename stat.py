from __future__ import print_function
import argparse
import numpy as np
import pickle as pkl


def f_score(label, predict):
    t_label = label[0]
    t_predict = predict[0]
    for i in range(len(label)):
        t_label += label[i]
        t_predict += predict[i]
    t_label = np.array(t_label > 0)
    t_predict = np.array(t_predict > 0)
    precision = np.sum(t_label * t_predict, axis=0) / np.sum(t_predict, axis=0)
    recall = np.sum(t_label * t_predict, axis=0) / np.sum(t_label, axis=0)
    f1 = 2 * precision * recall / (precision + recall)
    print("class wise stats:")
    print(precision)
    print(recall)
    print(f1)
    total_precision = np.sum(t_label * t_predict) / np.sum(t_predict)
    total_recall = np.sum(t_label * t_predict) / np.sum(t_label)
    total_f1 = 2 * total_precision * total_recall / (total_precision + total_recall)
    print("total stats")
    print(total_precision)
    print(total_recall)
    print(total_f1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--files', type=str, nargs='+', help='path of consecutive labels and predictions',
                        required=True)
    args = parser.parse_args()
    assert len(args.files) % 2 == 0
    label = []
    predict = []
    for i in range(0, len(args.files), 2):
        label.append(pkl.load(open(args.files[i], 'rb')))
        predict.append(pkl.load(open(args.files[i + 1], 'rb')))
    f_score(label, predict)


if __name__ == '__main__':
    main()
