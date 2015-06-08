from sys import argv, stderr
import cPickle

from featurize import Tokenizer, Featurizer
from dsl.representation.model import Representation


def main():
    N = int(argv[1]) if len(argv) > 1 else 3
    t = Tokenizer()
    f = Featurizer(t, N=N)
    f.featurize_in_directory(argv[2])
    #m = f.to_dok_matrix(docs)
    f.get_correlations()
    f.label_feat_pearson()
    cut = int(argv[4]) if len(argv) > 4 else 40
    f.filter_top_ngrams(cut)
    f.save_features('train_features')
    mtx = f.to_dok_matrix()
    with open('train_mtx.cPickle', 'wb') as fh:
        cPickle.dump((f.labels.l, mtx), fh, -1)
    stderr.write('Data read\n')
    stderr.write('Trained\n')
    test_f = Featurizer(t, N=N)
    test_f.featdict = f.featdict
    test_f.featdict.freeze_dict()
    test_f.featurize_in_directory(argv[3])
    docs = test_f.filter_ngrams(test_f.docs, f.topngrams)
    test_f.docs = docs
    test_f.topngrams = f.topngrams
    test_f.save_features('test_features')
    test_f.featdict.save('topfeatures')
    return
    test_mtx = test_f.to_dok_matrix()
    with open('test_mtx.cPickle', 'wb') as fh:
        cPickle.dump((test_f.labels.l, test_mtx), fh, -1)
    acc = 0
    stderr.write('Test matrix done\n')
    r = Representation('dummy', 'svm', svm_ktype='svc')
    r.encode(mtx)
    stderr.write('Encoded\n')
    r.train_classifier(f.labels.l)
    for i in xrange(test_mtx.shape[0]):
        gold = test_f.labels.l[i]
        cl = r.classify_vector(test_mtx.getrow(i).todense())[0]
        if gold == cl:
            acc += 1
    print float(acc) / test_mtx.shape[0]


if __name__ == '__main__':
    main()
