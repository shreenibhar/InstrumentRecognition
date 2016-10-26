import argparse
import os


def process_class(master, key):
    count = 0
    for audio_name in os.listdir(master):
        origin_path = os.path.join(master, audio_name)
        rename_path = os.path.join(master, key + '_' + str(count) + '.wav')
        os.rename(origin_path, rename_path)
        count += 1


parser = argparse.ArgumentParser()
parser.add_argument('--data_set', type=str, help='data set path',
                    required=True)
args = parser.parse_args()
for name in os.listdir(args.irmas_data_set):
    master_path = os.path.join(args.irmas_data_set, name)
    process_class(master_path, name)
    os.system('mv ' + os.path.join(master_path, '*') + ' ' +
              args.irmas_data_set)
    os.system('rm -r ' + master_path)
