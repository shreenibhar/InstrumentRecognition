from __future__ import print_function

import argparse
import os
import pickle as pkl

import numpy as np
from python_speech_features import mfcc
from scipy.io import wavfile


class Data:
    def __init__(self, data_dir, segment_size):
        self.data_dir = data_dir
        self.segment_size = segment_size
        self.input_data = []
        self.output_data = []

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

    def process_audio(self, audio_name):
        path = os.path.join(self.data_dir, audio_name)
        fs, audio_data = wavfile.read(path)
        assert fs == 44100
        if len(audio_data.shape) > 1:
            audio_data = np.sum(audio_data, axis=1) / audio_data.shape[1]
        audio_data = self.zero_trim(audio_data)
        num_segments = (audio_data.size + self.segment_size - 1) // self.segment_size
        audio_data = np.resize(audio_data, (num_segments, self.segment_size))
        mfcc_list = []
        for segment in audio_data:
            mfcc_list.append(mfcc(segment, fs))
        return mfcc_list

    @staticmethod
    def process_label(label_map, audio_name):
        for label in label_map:
            if label in audio_name:
                return label_map[label]
        return len(label_map)

    def process_dir(self, label_map):
        input_data = []
        output_data = []
        counter = 1
        for audio_name in os.listdir(self.data_dir):
            _ = self.process_audio(audio_name)
            label = self.process_label(label_map, audio_name)
            label = [label] * len(_)
            input_data.extend(_)
            output_data.extend(label)
            print('\rprocessed %f' % (counter * 100 / len(os.listdir(self.data_dir))), end='')
            counter += 1
        print()
        self.input_data = np.array(input_data)
        self.output_data = np.array(output_data)

    def save(self, data_path, label_path):
        assert len(self.input_data) == len(self.output_data)
        assert len(self.input_data) != 0
        pkl.dump(self.input_data, open(data_path, 'wb'))
        pkl.dump(self.output_data, open(label_path, 'wb'))

    def load(self, data_path, label_path):
        self.input_data = pkl.load(open(data_path, 'rb'))
        self.output_data = pkl.load(open(label_path, 'rb'))
        shuffle_index = np.random.permutation(len(self.input_data))
        self.input_data, self.output_data = self.input_data[shuffle_index], self.output_data[shuffle_index]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_set_dir', type=str, help='data set directory', required=True)
    parser.add_argument('--segment_size', type=int, help='segment size', required=True)
    parser.add_argument('--labels', type=str, nargs='+', help='class labels', required=True)
    parser.add_argument('--data_set_save', type=str, nargs=2, help='paths for data and labels to be saved',
                        required=True)
    args = parser.parse_args()
    label_map = {}
    for label in args.labels:
        label_map[label] = len(label_map)
    data_set = Data(args.data_set_dir, args.segment_size)
    data_set.process_dir(label_map)
    data_set.save(args.data_set_save[0], args.data_set_save[1])


if __name__ == '__main__':
    main()
