from sys import argv
import logging
import numpy as np


def pearson_mtx(mtx1, mtx2):
    assert mtx1.shape[0] == mtx2.shape[0]
    if len(mtx1.shape) == 1:
        mtx1 = np.reshape(mtx1, (mtx1.shape[0], 1))
    if len(mtx2.shape) == 1:
        mtx2 = np.reshape(mtx2, (mtx2.shape[0], 1))
    logging.info('Matrix1 size: {}\nMatrix2 size: {}'.format(mtx1.shape, mtx2.shape))
    n = mtx1.shape[0]
    u = mtx1.shape[1]
    v = mtx2.shape[1]
    means1 = mtx1.mean(0)
    logging.info('Means 1 computed: {}'.format(means1.shape))
    means2 = mtx2.mean(0)
    logging.info('Means 2 computed: {}'.format(means2.shape))
    sqsum1 = mtx1.transpose().dot(mtx1).diagonal()
    logging.info('SqMeans 1 computed: {}'.format(sqsum1.shape))
    sqsum2 = mtx2.transpose().dot(mtx2).diagonal()
    logging.info('SqMeans 2 computed: {}'.format(sqsum2.shape))
    pearson = np.zeros((u, v), np.float)
    x_ = np.sqrt(np.array([sqsum1[i] - n * (means1[0, i] ** 2) for i in xrange(u)], dtype=np.float64))
    y_ = np.sqrt(np.array([sqsum2[i] - n * (means2[0, i] ** 2) for i in xrange(v)], dtype=np.float64))
    for i in xrange(u):
        vec1 = mtx1[:, i]
        for j in xrange(v):
            vec2 = mtx2[:, j]
            s = vec1.transpose() * vec2
            s = s.sum() - means1[0, i] * means2[0, j] * n
            if x_[i] == 0 or y_[j] == 0:
                pearson[i, j] = 0
            else:
                pearson[i, j] = s / (x_[i] * y_[j])
    return pearson


def main():
    with open(argv[1]) as f:
        mtx1 = np.loadtxt(f, np.float)
    with open(argv[2]) as f:
        mtx2 = np.loadtxt(f, np.float)
    print pearson_mtx(mtx1, mtx2)


if __name__ == '__main__':
    main()
