from argparse import ArgumentParser
from collections import defaultdict
from sys import stderr
import math
import os


def parse_args():
    p = ArgumentParser()
    p.add_argument('--train', type=str, help='train dir')
    p.add_argument('--test', type=str, help='test dir')
    p.add_argument('--tf', choices=['raw', 'binary', 'lognorm'], default='raw')
    p.add_argument('--idf', choices=['invfreq', 'smooth'], default='invfreq')
    p.add_argument('--topn', type=int, default=100)
    p.add_argument('--rare', type=int, default=5)
    p.add_argument('--N', type=int, default=3)
    p.add_argument('--tokenize', choices=['ngram', 'word', 'mixed'], default='word')
    return p.parse_args()


def trim_word(word):
    return word.strip()


def tokenize(doc, args):
    if args.tokenize == 'word' or args.tokenize == 'mixed':
        for word in doc.split():
            tr = trim_word(word)
            if tr:
                yield tr
    if args.tokenize == 'ngram' or args.tokenize == 'mixed':
        N = args.N
        for i in xrange(0, len(doc) - N + 1):
            yield doc[i:i + N]


def tf(doc_stream, args):
    mode = args.tf
    rare = args.rare
    if mode == 'raw':
        tf = tf_rawfreq(doc_stream, args)
    if mode == 'binary':
        tf = tf_binary(doc_stream, args)
    if mode == 'lognorm':
        tf = tf_lognorm(doc_stream, args)
    if rare > 0:
        tf_freq = {}
        for w, cnt in tf.iteritems():
            if cnt >= rare:
                tf_freq[w] = cnt
        return tf_freq
    return tf


def tf_lognorm(doc_stream, args):
    tf = defaultdict(int)
    for line in doc_stream:
        for word in tokenize(line.decode('utf8').strip(), args):
            trimmed = trim_word(word)
            if trimmed:
                tf[trimmed] += 1
    tf_log = {}
    for term, cnt in tf.iteritems():
        tf_log[term] = math.log(1 + cnt)
    return tf_log


def tf_binary(doc_stream, args):
    tf = defaultdict(int)
    for line in doc_stream:
        for word in tokenize(line.decode('utf8').strip(), args):
            trimmed = trim_word(word)
            if trimmed:
                tf[trimmed] = 1
    return tf


def tf_rawfreq(doc_stream, args):
    tf = defaultdict(int)
    for line in doc_stream:
        for word in tokenize(line.decode('utf8').strip(), args):
            trimmed = trim_word(word)
            if trimmed:
                tf[trimmed] += 1
    return tf


def idf(doc_tfs, mode):
    if mode == 'invfreq':
        return idf_invfreq(doc_tfs)
    if mode == 'smooth':
        return idf_smooth(doc_tfs)


def idf_smooth(doc_tfs):
    idf_cnt = defaultdict(int)
    all_terms = set()
    for doc, tf in doc_tfs.iteritems():
        all_terms |= set(tf.iterkeys())
    for term in all_terms:
        for tf in doc_tfs.itervalues():
            if term in tf:
                idf_cnt[term] += 1
    idf = {}
    N = len(doc_tfs)
    for term, cnt in idf_cnt.iteritems():
        idf[term] = math.log(1 + N / float(cnt))
    return idf


def idf_invfreq(doc_tfs):
    idf_cnt = defaultdict(int)
    all_terms = set()
    for doc, tf in doc_tfs.iteritems():
        all_terms |= set(tf.iterkeys())
    for term in all_terms:
        for tf in doc_tfs.itervalues():
            if term in tf:
                idf_cnt[term] += 1
    idf = {}
    N = len(doc_tfs)
    for term, cnt in idf_cnt.iteritems():
        idf[term] = math.log(N / float(cnt))
    return idf


def tfidf(doc_tfs, idf, top=10):
    tfidf = {}
    for doc, tf in doc_tfs.iteritems():
        tfidf[doc] = {}
        for term, cnt in tf.iteritems():
            tfidf[doc][term] = cnt * idf[term]
    tfidf_top = {}
    for doc, terms in tfidf.iteritems():
        tfidf_top[doc] = {}
        n = top
        for term, score in sorted(terms.iteritems(), key=lambda x: -x[1])[:top]:
            tfidf_top[doc][term] = n
            n -= 1
    return tfidf_top


def train(args):
    dir_ = args.train
    doc_tfs = {}
    for fn in os.listdir(dir_):
        with open(os.path.join(dir_, fn)) as f:
            doc_tfs[fn] = tf(f, args)
    idf_d = idf(doc_tfs, args.idf)
    return tfidf(doc_tfs, idf_d, top=args.topn), idf_d


def test(tfidf, idf, args):
    results = defaultdict(lambda: defaultdict(int))
    for fn in os.listdir(args.test):
        with open(os.path.join(args.test, fn)) as f:
            for doc in f:
                guess = classify_text(doc.decode('utf8'), tfidf, idf)
                results[fn][guess] += 1
    return results


def classify_text(doc, tfidf, idf):
    hits = defaultdict(int)
    test_terms = get_test_terms(doc, idf)
    for term, weight in test_terms.iteritems():
        tr = trim_word(term)
        if not tr.strip():
            continue
        for lang, terms in tfidf.iteritems():
            if term in terms:
                hits[lang] += terms[term] * weight
    if not hits:
        return 'UN'
    return max(hits.iteritems(), key=lambda x: x[1])[0]


def get_test_terms(doc, idf):
    words = defaultdict(int)
    for w in doc.split():
        tr = trim_word(w)
        if tr:
            words[tr] += 1
    weights = {}
    max_f = max(words.iteritems(), key=lambda x: x[1])[1]
    for word, f in words.iteritems():
        if not word in idf:
            continue
        weights[word] = (0.5 + 0.5 * float(f) / max_f) * idf[word]
    return weights


def print_stats(t):
    good = 0
    N = 0
    for tr, guesses in t.iteritems():
        good += guesses.get(tr, 0)
        N += sum(guesses.itervalues())
    t['acc'] = float(good) / N
    import json
    print json.dumps(t)


def main():
    args = parse_args()
    tfidf, idf = train(args)
    stderr.write('Trained\n')
#    for lang, terms in topn.iteritems():
#        for term, score in terms:
#            print(u'{0}\t{1}\t{2}'.format(lang, term, score).encode('utf8'))
    t = test(tfidf, idf, args)
    print_stats(t)

if __name__ == '__main__':
    main()
