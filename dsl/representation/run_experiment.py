from sys import stderr, stdin
from os import listdir
from model import Representation
from argparse import ArgumentParser
from sklearn.preprocessing import scale
from scipy.io import mmread
import numpy as np
import logging


def parse_args():
    p = ArgumentParser()
    p.add_argument('-e', '--encoder', type=str, default='dummy')
    p.add_argument('-c', '--classifier', type=str, default='simple')
    p.add_argument('params', nargs='*')
    p.add_argument('--lang-map', type=str, default='dat/t1/train')
    p.add_argument('--scale', action='store_true', default=False)
    p.add_argument('--with-probs', action='store_true', default=False)
    p.add_argument('--train', type=str)
    p.add_argument('--test', type=str)
    return p.parse_args()


def main():
    FORMAT = '%(asctime)s %(levelname)s %(message)s'
    logging.basicConfig(format=FORMAT)
    logging.getLogger().setLevel(logging.INFO)
    args = parse_args()
    lang_map = {i: fn for i, fn in enumerate(sorted(listdir(args.lang_map)))}
    if args.train.endswith('.mtx'):
        mtx = mmread(args.train).todense()
        t_mtx = mmread(args.test).todense()
    else:
        with open(args.train) as stream:
            mtx = np.loadtxt(stream, np.float64)
        with open(args.test) as stream:
            t_mtx = np.loadtxt(stream, np.float64)
    labels = np.ravel(mtx[:, 0])
    test_labels = t_mtx[:, 0]
    test_mtx = t_mtx[:, 1:]
    if args.scale:
        train = scale(mtx[:, 1:], with_mean=False)
    else:
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
    for vec_ in test_mtx:
        vec = np.ravel(vec_)
        cl = r.classify_vector(vec, with_probs=args.with_probs)
        try:
            lab = test_labels[N, 0]
        except IndexError:
            lab = test_labels[N]
        N += 1
        if args.with_probs:
            guess = max(enumerate(cl[0, :]), key=lambda x: x[1])[0]
            print('{0}\t{1}\t{2}'.format('\t'.join(map(str, cl[0, :])), lang_map[guess], lang_map[int(lab)]))
        else:
            try:
                guess = int(cl[0, 0])
            except IndexError:
                guess = int(cl + 0.5)
            print('{0}\t{1}'.format(lang_map[guess], lang_map[int(lab)]))
        if int(guess) == int(lab):
            acc += 1
    #print(float(acc) / N)

if __name__ == '__main__':
    main()
