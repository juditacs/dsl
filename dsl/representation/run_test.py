from sys import stderr, stdin, argv
from model import Representation
import numpy as np


def main():
    lang_map = {
        0: 'bg',
        1: 'bs',
        2: 'cz',
        3: 'es-AR',
        4: 'es-ES',
        5: 'hr',
        6: 'id',
        7: 'mk',
        8: 'my',
        9: 'pt-BR',
        10: 'pt-PT',
        11: 'sk',
        12: 'sr',
        13: 'xx',
    }
#    lang_map = {
#        0: 'bs',
#        1: 'cz',
#        2: 'hr',
#        3: 'sk',
#        4: 'sr',
#    }
    lang_map = {
        0: 'bs',
        1: 'hr',
        2: 'sr',
    }
    lang_map = {
        0: 'pt-BR',
        1: 'pt-PT',
    }
    mtx = np.loadtxt(stdin, np.float)
    labels = mtx[:, 0]
    train = mtx[:, 1:]
    r = Representation('dummy', 'svm', rbm_hiddenDimension=10, simple_field=3, pca_dimension=50, svm_ktype='svc', svm_kernel='linear')
    r.encode(train)
    stderr.write('Encoded\n')
    r.train_classifier(labels)
    stderr.write('Trained\n')
    acc = 0
    N = 0
    with open(argv[1]) as f:
        for l in f:
            fs = l.strip().split()
            lab = int(fs[0])
            vec = np.array(map(float, fs[1:]))
            cl = r.classify_vector(vec)
            try:
                guess = int(cl[0])
            except IndexError:
                guess = int(cl + 0.5)
            N += 1
            if int(guess) == int(lab):
                acc += 1
            print('{0}\t{1}'.format(lang_map[guess], lang_map[int(lab)]))
    print(float(acc) / N)

if __name__ == '__main__':
    main()
