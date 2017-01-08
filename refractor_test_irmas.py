import os
import argparse


def shell_quote(s):
    return "'" + s.replace("'", "'\\''") + "'"


parser = argparse.ArgumentParser()
parser.add_argument('--data_set', type=str, help='data set path', required=True)
args = parser.parse_args()
unique_dict = {}
for f in os.listdir(args.data_set):
    name = '.'.join(f.split('.')[0:-1])
    ext = str(f.split('.')[-1])
    if ext == 'wav':
        label = open(os.path.join(args.data_set, name + '.txt')).read().split('\n')
        label = [str(val.strip()) for val in label if len(str(val.strip())) != 0]
        label = '_'.join(label)
        no = unique_dict.get(label, 0)
        os.rename(os.path.join(args.data_set, f), os.path.join(args.data_set, label + '_' + str(no) + '.wav'))
        os.remove(os.path.join(args.data_set, name + '.txt'))
        unique_dict[label] = no + 1
