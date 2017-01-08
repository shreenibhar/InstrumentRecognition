import os
import argparse
import numpy as np
import pickle as pkl


def combine(directory, location, start, end):
    for name in os.listdir(directory):
        if name[location] == str(start):
            combined_shape = list(pkl.load(open(os.path.join(directory, name), 'rb')).shape)
            combined_shape[0] = 0
            combined = np.zeros(combined_shape)
            for i in range(start, end + 1):
                rename = name.replace(str(start), str(i))
                rename_file = pkl.load(open(os.path.join(directory, rename), 'rb'))
                combined = np.append(combined, rename_file, axis=0)
                os.system('rm ' + os.path.join(directory, rename))
            rename = name.replace(str(start), "")
            pkl.dump(combined, open(os.path.join(directory, rename), 'wb'))


parser = argparse.ArgumentParser()
parser.add_argument('--directory', type=str, help='path of predict path', required=True)
parser.add_argument('--location', type=int, help='index location of the test set number in the file names',
                    required=True)
parser.add_argument('--start', type=int, help='start number of test set', required=True)
parser.add_argument('--end', type=int, help='end number of test set', required=True)
args = parser.parse_args()
combine(args.directory, args.location, args.start, args.end)
