from sys import argv

from featurize import Tokenizer, Featurizer


def main():
    N = int(argv[1]) if len(argv) > 1 else 3
    t = Tokenizer()
    f = Featurizer(t, N=N)
    docs = f.featurize_in_directory(argv[2])
    m = f.to_dok_matrix(docs)
    print m.shape

if __name__ == '__main__':
    main()
