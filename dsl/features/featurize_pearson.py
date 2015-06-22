from sys import stderr
from os import path
from argparse import ArgumentParser
import logging

from featurize import Tokenizer, Featurizer, CharacterNgram


def parse_args():
    p = ArgumentParser()
    p.add_argument('train', type=str)
    p.add_argument('--train-mtx', type=str)
    p.add_argument('test', type=str)
    p.add_argument('--test-mtx', type=str)
    p.add_argument('--pearson',  help='write correlations to file', type=str, default='')
    p.add_argument('-N', '--N', type=int, default=2)
    p.add_argument('-t', '--threshold', type=int, default=2)
    p.add_argument('--topn', type=int, default=200)
    p.add_argument('--train-out', type=str)
    p.add_argument('--test-out', type=str)
    p.add_argument('--strategy', choices=['simmple', 'diff'], default='simple')
    return p.parse_args()


def main():
    args = parse_args()
    if (args.train_mtx and not args.test_mtx) or (args.test_mtx and not args.train_mtx):
        stderr.write('Please specify both --train-mtx and --test-mtx or neither.')
        return
    FORMAT = '%(asctime)s %(levelname)s %(message)s'
    logging.basicConfig(format=FORMAT)
    logging.getLogger().setLevel(logging.INFO)
    N = args.N
    t = Tokenizer(filter_punct=True, ws_norm=True, strip=True, replace_digits=True)
    train_f = Featurizer(t, N=N)
    if args.train_mtx and path.exists(args.train_mtx):
        train_f.load_matrix(args.train_mtx)
    else:
        ngrams = CharacterNgram(t)
        ngrams.count_in_directory(args.train, N=N, padding=False)
        ngram_filt = ngrams.get_frequent(threshold=args.threshold)
        train_f.featurize_in_directory(args.train, feature_filt=ngram_filt)
        logging.info('Featurized')
        #m = f.to_dok_matrix(docs)
        #train_mtx = f.to_dok_matrix()
        train_f.choose_top_pearson(args.topn, strategy=args.strategy)
        logging.info('Top {} pearson chosen'.format(args.topn))
        with open(args.train_out, 'w') as f:
            train_f.save_features(f)
        if args.train_mtx:
            train_f.save_matrix(args.train_mtx)
        train_f.featdict.freeze_dict()
    test_f = Featurizer(t, N=N)
    if args.test_mtx and path.exists(args.test_mtx):
        test_f.load_matrix(args.test_mtx)
    else:
        test_f.featdict = train_f.featdict
        test_f.featurize_in_directory(args.test, feature_filt=ngram_filt)
        test_f.topngrams = train_f.topngrams
        with open(args.test_out, 'w') as f:
            test_f.save_features(f)
        if args.test_mtx:
            test_f.save_matrix(args.test_mtx)
    if args.pearson:
        train_f.write_pearson(args.pearson, filter_top=True)


if __name__ == '__main__':
    main()
    #import cProfile
    #cProfile.run('main()', 'pearson.stats')
