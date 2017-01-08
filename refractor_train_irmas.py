import os
import argparse


def class_finder(audio_name):
    if "cel" in audio_name:
        return "cel"
    if "flu" in audio_name:
        return "flu"
    if "gac" in audio_name:
        return "gac"
    if "gel" in audio_name:
        return "gel"
    if "org" in audio_name:
        return "org"
    if "pia" in audio_name:
        return "pia"
    if "sax" in audio_name:
        return "sax"
    if "tru" in audio_name:
        return "tru"
    if "vio" in audio_name:
        return "vio"
    if "voi" in audio_name:
        return "voi"
    if "cla" in audio_name:
        return "cla"


parser = argparse.ArgumentParser()
parser.add_argument('--data_set', type=str, help='data set path',
                    required=True)
args = parser.parse_args()
class_dict = {}
for name in os.listdir(args.data_set):
    l = class_dict.get(class_finder(name), 0)
    os.system(
        'mv ' + os.path.join(args.data_set, name) + ' ' + os.path.join(args.data_set,
                                                                       class_finder(name) + '_%d.wav' % l))
    class_dict[class_finder(name)] = l + 1
