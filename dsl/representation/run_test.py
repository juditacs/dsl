from sys import stderr, stdin
from os import listdir
from model import Representation
from argparse import ArgumentParser
import numpy as np
import logging


def parse_args():
    p = ArgumentParser()
    p.add_argument('-e', '--encoder', type=str, default='dummy')
    p.add_argument('-c', '--classifier', type=str, default='simple')
    p.add_argument('params', nargs='*')
    p.add_argument('--lang-map', type=str, default='dat/t1/train')
    p.add_argument('--train', type=str)
    p.add_argument('--test', type=str)
    return p.parse_args()


def main():
    FORMAT = '%(asctime)s %(levelname)s %(message)s'
    logging.basicConfig(format=FORMAT)
    logging.getLogger().setLevel(logging.INFO)
    args = parse_args()
    lang_map = {i: fn for i, fn in enumerate(sorted(listdir(args.lang_map)))}
    with open(args.train) as stream:
        mtx = np.loadtxt(stream, np.int16)
    labels = mtx[:, 0]
    train = mtx[:, 1:]
    kwargs = {}
    for a in args.params:
        k, v = a.split('=')
        try:
            v = int(v)
        except:
            pass
        kwargs[k] = v
    r = Representation(args.encoder, args.classifier, **kwargs)
    r.encode(train)
    logging.info('Matrix encoded')
    r.train_classifier(labels)
    logging.info('Model trained')
    acc = 0
    N = 0
    with open(args.test) as f:
        for l in f:
            fs = l.strip().split()
            lab = int(fs[0])
            vec = np.array(map(float, fs[1:]))
            cl = r.classify_vector(vec)
            try:
                guess = int(cl[0])
            except IndexError:
                guess = int(cl + 0.5)
            N += 1
            if int(guess) == int(lab):
                acc += 1
            print('{0}\t{1}'.format(lang_map[guess], lang_map[int(lab)]))
    print(float(acc) / N)

if __name__ == '__main__':
    main()
