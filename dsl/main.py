from argparse import ArgumentParser
from ConfigParser import ConfigParser
from sentence import SentenceCollection


def parse_args():
    p = ArgumentParser()
    p.add_argument('--train', help='train directory', type=str, default='train')
    p.add_argument('--dev', help='dev directory', type=str)
    p.add_argument('--test', help='test directory', type=str)
    p.add_argument('-c', '--config', help='config file', default='cfg/default.cfg', type=str)
    return p.parse_args()


def main():
    config = ConfigParser()
    config.readfp(open(args.config))
    train = SentenceCollection(config)
    train.read_sentences(args.train, labeled=True)
    train.featurize()
    if args.dev:
        dev = SentenceCollection()
        dev.read_sentences(args.dev, labeled=True)
        dev.featurize()
    if args.test:
        test = SentenceCollection()
        test.read_sentences(args.test, labeled=True)
        test.featurize()


if __name__ == '__main__':
    args = parse_args()
    main()
