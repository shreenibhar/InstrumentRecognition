from __future__ import print_function
import os
import argparse
import numpy as np
import pickle as pkl
from scipy.io import wavfile
from python_speech_features import mfcc


class Data:
    def __init__(self, data_dir, segment_size):
        self.data_dir = data_dir
        self.segment_size = segment_size
        self.input_data = []
        self.output_data = []
        self.cluster = []

    @staticmethod
    def zero_trim(arr):
        st = -1
        ed = -1
        for i in range(len(arr)):
            if i != 0:
                ed = i
                if st == -1:
                    st = i
        return arr[st:ed + 1]

    def process_audio(self, audio_name, strip_zeros, scale, pan):
        path = os.path.join(self.data_dir, audio_name)
        fs, audio_data = wavfile.read(path)
        assert fs == 44100
        if pan == "n":
            audio_data = audio_data[:, 0] + audio_data[:, 1]
        elif pan == "l":
            audio_data = audio_data[:, 0]
        elif pan == "r":
            audio_data = audio_data[:, 1]
        elif pan == "s":
            audio_data = audio_data[:, 0] - audio_data[:, 1]
        if strip_zeros:
            audio_data = self.zero_trim(audio_data)
        num_segments = (audio_data.size + self.segment_size - 1) // self.segment_size
        resize_shape = (num_segments, self.segment_size)
        audio_data = np.resize(audio_data, resize_shape)
        if scale:
            audio_fft = np.fft.fft(audio_data, axis=1)
            scaler = np.sum(audio_fft.__abs__(), axis=1, keepdims=True)
            nonzero = np.nonzero(scaler)[0]
            audio_fft, scaler = audio_fft[nonzero, :], scaler[nonzero]
            audio_fft /= scaler
            audio_data = np.fft.ifft(audio_fft, axis=1).real
        mfcc_list = []
        win_len = .025
        win_step = .01
        n_fft = int(win_len * fs) + 1
        for segment in audio_data:
            mfcc_list.append(mfcc(segment, samplerate=fs, winlen=win_len, winstep=win_step, nfft=n_fft))
        return mfcc_list

    @staticmethod
    def process_label(label_map, audio_name, option):
        if option == 'single':
            for label in label_map:
                if label in audio_name:
                    return label_map[label]
            return len(label_map) - 1
        elif option == 'multiple':
            label_array = np.zeros(len(label_map))
            for label in label_map:
                if label in audio_name:
                    label_array[label_map[label]] = 1
            if np.sum(label_array) == 0:
                label_array[-1] = 1
            return label_array

    def process_dir(self, label_map, strip_zeros, scale, label_option, pan):
        input_data = []
        output_data = []
        cluster = []
        counter = 1
        for audio_name in os.listdir(self.data_dir):
            _ = self.process_audio(audio_name, strip_zeros=strip_zeros, scale=scale, pan=pan)
            label = self.process_label(label_map, audio_name, option=label_option)
            label = [label] * len(_)
            input_data.extend(_)
            output_data.extend(label)
            cluster.append(len(_))
            print('\rprocessed %f' % (counter * 100 / len(os.listdir(self.data_dir))), end='')
            counter += 1
        print()
        self.input_data = np.array(input_data)
        self.output_data = np.array(output_data)
        self.cluster = cluster

    def save(self, data_path, label_path, cluster_path):
        assert len(self.input_data) == len(self.output_data) == np.sum(self.cluster)
        pkl.dump(self.input_data, open(data_path, 'wb'))
        pkl.dump(self.output_data, open(label_path, 'wb'))
        pkl.dump(self.cluster, open(cluster_path, 'wb'))

    def load(self, data_path, label_path, cluster_path):
        self.input_data = pkl.load(open(data_path, 'rb'))
        self.output_data = pkl.load(open(label_path, 'rb'))
        self.cluster = pkl.load(open(cluster_path, 'rb'))
        assert len(self.input_data) == len(self.output_data) == np.sum(self.cluster)

    def shuffle(self):
        shuffle_index = np.random.permutation(len(self.input_data))
        self.input_data, self.output_data = self.input_data[shuffle_index], self.output_data[shuffle_index]

    def divide_cluster(self, percentage):
        new_data_set = Data('', 22050)
        boundary = int(percentage * len(self.input_data) / 100)
        start = 0
        index = 0
        for no in self.cluster:
            if start >= boundary:
                break
            start += no
            index += 1
        boundary = start
        new_data_set.input_data, new_data_set.output_data = self.input_data[boundary:], self.output_data[boundary:]
        new_data_set.cluster = self.cluster[index:]
        self.input_data, self.output_data = self.input_data[:boundary], self.output_data[:boundary]
        self.cluster = self.cluster[:index]
        assert len(self.input_data) == len(self.output_data) == np.sum(self.cluster)
        assert len(new_data_set.input_data) == len(new_data_set.output_data) == np.sum(new_data_set.cluster)
        return new_data_set


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_set_dir', type=str, help='data set directory', required=True)
    parser.add_argument('--segment_size', type=int, help='segment size', required=True)
    parser.add_argument('--labels', type=str, nargs='+', help='class labels including null class', required=True)
    parser.add_argument('--strip_zeros', type=bool, help='strip zeros at end option', required=True)
    parser.add_argument('--scale', type=bool, help='scaling enabler', required=True)
    parser.add_argument('--pan', type=str, help='pan type n/l/r/s', required=True)
    parser.add_argument('--label_option', type=str, help='label type single or multiple', required=True)
    parser.add_argument('--data_set_save', type=str, nargs=3,
                        help='paths for data and labels and cluster in each sample',
                        required=True)
    args = parser.parse_args()
    assert args.label_option in ['single', 'multiple']
    assert args.pan in ['n', 'l', 'r', 's']
    label_map = {}
    for label in args.labels:
        label_map[label] = len(label_map)
    data_set = Data(args.data_set_dir, args.segment_size)
    data_set.process_dir(label_map, strip_zeros=args.strip_zeros, scale=args.scale,
                         label_option=args.label_option, pan=args.pan)
    data_set.save(args.data_set_save[0], args.data_set_save[1], args.data_set_save[2])


if __name__ == '__main__':
    main()
