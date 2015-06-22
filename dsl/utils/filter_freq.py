from os import listdir, path
from argparse import ArgumentParser
from collections import defaultdict

from dsl.features.featurize import Tokenizer, WordNgram


def parse_args():
    p = ArgumentParser()
    p.add_argument('--train', type=str)
    p.add_argument('--train-out', type=str)
    p.add_argument('--test', type=str)
    p.add_argument('--test-out', type=str)
    p.add_argument('--topn', type=int, default=100)
    p.add_argument('--N', type=int, default=1, help='N of wordngram counter')
    p.add_argument('--mode', choices=['keep', 'remove'], default='keep')
    return p.parse_args()


def read_topn_in_dir(basedir, counters, topn=100):
    topwords = {}
    for fn in sorted(listdir(basedir)):
        with open(path.join(basedir, fn)) as f:
            counters[fn].count_words_in_stream(f)
            topwords[fn] = counters[fn].get_topn_words(topn)
    return topwords


def filter_dir_to_top(indir, outdir, topwords, tokenizer):
    for fn in sorted(listdir(indir)):
        infile = open(path.join(indir, fn))
        outfile = open(path.join(outdir, fn), 'w')
        for sentence in tokenizer.tokenize_stream(infile):
            outfile.write(' '.join(filter(lambda x: x in topwords[fn], sentence)).encode('utf8') + '\n')
        infile.close()
        outfile.close()


def main():
    args = parse_args()
    t = Tokenizer()
    counters = defaultdict(lambda: WordNgram(t, N=args.N))
    topn = read_topn_in_dir(args.train, counters, args.topn)
    filter_dir_to_top(args.train, args.train_out, topn, t)
    filter_dir_to_top(args.test, args.test_out, topn, t)


if __name__ == '__main__':
    main()
