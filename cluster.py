import argparse
import numpy as np
import pickle as pkl


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
    parser.add_argument('--data_set_load', type=str, nargs=2,
                        help='path of label and cluster in each sample',
                        required=True)
    parser.add_argument('--predict_path', type=str, help='path of prediction', required=True)
    parser.add_argument('--save_path', type=str, nargs=2, help='path of label and predict clustered', required=True)
    args = parser.parse_args()
    label = np.load(args.data_set_load[0])
    cluster = np.load(args.data_set_load[1])
    predict = np.load(args.predict_path)
    clustered_label, clustered_predict = cluster_predict(label, cluster, predict)
    pkl.dump(clustered_label, open(args.save_path[0], 'wb'))
    pkl.dump(clustered_predict, open(args.save_path[1], 'wb'))


if __name__ == '__main__':
    main()
