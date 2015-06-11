from sys import argv, stderr
import math
import numpy as np


def pearson_mtx(mtx1, mtx2):
    assert mtx1.shape[0] == mtx2.shape[0]
    if len(mtx1.shape) == 1:
        mtx1 = np.reshape(mtx1, (mtx1.shape[0], 1))
    if len(mtx2.shape) == 1:
        mtx2 = np.reshape(mtx2, (mtx2.shape[0], 1))
    stderr.write('Matrix1 size: {}\nMatrix2 size: {}\n'.format(mtx1.shape, mtx2.shape))
    n = mtx1.shape[1]
    m = mtx2.shape[1]
    means1 = np.array([np.sum(mtx1[:, i]) / mtx1.shape[0] for i in xrange(n)])
    means2 = np.array([np.sum(mtx2[:, i]) / mtx2.shape[0] for i in xrange(m)])
    stderr.write('Means computed\n')
    sqsum1 = [np.dot(mtx1[:, i].transpose(), mtx1[:, i]) for i in xrange(n)]
    sqsum2 = [np.dot(mtx2[:, i].transpose(), mtx2[:, i]) for i in xrange(m)]
    stderr.write('Sqmeans computed\n')
    pearson = np.zeros((n, m), np.float)
    for i in xrange(n):
        vec1 = mtx1[:, i].transpose()
        for j in xrange(m):
            vec2 = mtx2[:, j]
            pearson[i, j] = (np.dot(vec1, vec2) - n * means1[i] * means2[j]) / (math.sqrt((sqsum1[i] - n * means1[i] ** 2) * (sqsum2[j] - n * means2[j] ** 2)))
    return pearson


def main():
    with open(argv[1]) as f:
        mtx1 = np.loadtxt(f, np.float)
    with open(argv[2]) as f:
        mtx2 = np.loadtxt(f, np.float)
    print pearson_mtx(mtx1, mtx2)


if __name__ == '__main__':
    main()
