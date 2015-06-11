from sys import stderr
from argparse import ArgumentParser

from featurize import Tokenizer, Featurizer, CharacterNgram


def parse_args():
    p = ArgumentParser()
    p.add_argument('train', type=str)
    p.add_argument('test', type=str)
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
    N = args.N
    t = Tokenizer(filter_punct=True, ws_norm=True, strip=True, replace_digits=True)
    train_f = Featurizer(t, N=N)
    ngrams = CharacterNgram(t)
    ngrams.count_in_directory(args.train, N=N, padding=False)
    ngram_filt = ngrams.get_frequent(threshold=args.threshold)
    train_f.featurize_in_directory(args.train, feature_filt=ngram_filt)
    stderr.write('Featurized\n')
    #m = f.to_dok_matrix(docs)
    #train_mtx = f.to_dok_matrix()
    stderr.write('Data read\n')
    train_f.choose_top_pearson(args.topn, strategy=args.strategy)
    with open(args.train_out, 'w') as f:
        train_f.save_features(f)
    train_f.featdict.freeze_dict()
    test_f = Featurizer(t, N=N)
    test_f.featdict = train_f.featdict
    test_f.featurize_in_directory(args.test, feature_filt=ngram_filt)
    test_f.topngrams = train_f.topngrams
    with open(args.test_out, 'w') as f:
        test_f.save_features(f)
    if args.pearson:
        train_f.write_pearson(args.pearson, filter_top=True)


if __name__ == '__main__':
    main()
