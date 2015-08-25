import numpy
from model import Representation


def main():
    r = Representation('pca',  'naive_bayes', dimension=3)
    raw_mtx = numpy.array([[1, 1, 1, 0, 0, 0], [1, 0, 1, 0, 0, 0], [1, 1, 1, 0, 0, 0], [0, 0, 1, 1, 1, 0],  [0, 0, 1, 1, 0, 0], [0, 0, 1, 1, 1, 0]])
    r.encode(raw_mtx)
    r.train_classifier([0, 0, 0, 1, 1, 1])
    print r.classify_vector([1, 2, 1, 0, 1, 0])

if __name__ == '__main__':
    main()
