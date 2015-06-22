import logging
from os import path
from argparse import ArgumentParser
from featurize import Tokenizer, BigramModel


def parse_args():
    p = ArgumentParser()
    p.add_argument('--train', type=str)
    p.add_argument('--test', type=str)
    p.add_argument('--raw-matrix-dir', type=str)
    p.add_argument('--workdir', type=str)
    p.add_argument('--topn', type=int, default=100)
    p.add_argument('--threshold', type=int, default=2)
    return p.parse_args()


def get_paths(base, workdir):
    raw_paths = [
        'train_raw.mtx',
        'test_raw.mtx',
        'train.labels',
        'train.labels.int',
        'test.labels',
        'test.labels.int',
        'frequent_features',
        'labeldict',
        'featdict',
    ]
    workdir_paths = [
        'top_corr_features',
        'train_top_corr.mtx',
        'test_top_corr.mtx',
        'train_dense.mtx',
        'test_dense.mtx',
    ]
    paths = {k: path.join(base, k) for k in raw_paths}
    paths.update({k: path.join(workdir, k) for k in workdir_paths})
    return paths


def main():
    FORMAT = '%(asctime)s %(levelname)s %(message)s'
    logging.basicConfig(format=FORMAT)
    logging.getLogger().setLevel(logging.DEBUG)
    args = parse_args()
    paths = get_paths(args.raw_matrix_dir, args.workdir)
    t = Tokenizer(ws_norm=True, filter_punct=True)
    b_train = BigramModel(t, padding=True)
    b_train.paths = paths
    b_train.load_or_build_train(args.train, args.threshold)
    b_train.choose_top_pearson(args.topn)
    b_train.save_top_corr_features()
    b_train.to_filtered_matrix()
    b_train.save_matrix(paths['train_top_corr.mtx'])
    b_train.save_as_dense_matrix(paths['train_dense.mtx'])

    b_train.load_or_build_test(args.test)
    b_train.to_filtered_matrix()
    b_train.save_matrix(paths['test_top_corr.mtx'])
    b_train.save_as_dense_matrix(paths['test_dense.mtx'])

if __name__ == '__main__':
    main()
