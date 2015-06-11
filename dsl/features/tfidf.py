# -*- coding: utf-8 -*-
from argparse import ArgumentParser
from collections import defaultdict
from sys import stderr
from nltk.tokenize import word_tokenize
import math
import os
import re
import string
import json

num_re = re.compile(r'^[0-9,.]+$', re.UNICODE)
punct_re = re.compile(ur'[{0}{1}]'.format(string.punctuation, '“”«»'.decode('utf8')), re.UNICODE)


def parse_args():
    p = ArgumentParser()
    p.add_argument('comment', nargs='?', type=str, default='')
    p.add_argument('--train', type=str, help='train dir')
    p.add_argument('--test', type=str, help='test dir')
    p.add_argument('--tf', choices=['raw', 'binary', 'lognorm'], default='raw')
    p.add_argument('--idf', choices=['invfreq', 'smooth'], default='invfreq')
    p.add_argument('--qf', choices=['raw', 'lognorm', 'smooth', 'smoothnorm'], default='smooth')
    p.add_argument('--topn', type=int, default=100)
    p.add_argument('--rare', type=int, default=5)
    p.add_argument('--N', type=int, default=3)
    p.add_argument('--lower', action='store_true', default=False)
    p.add_argument('--tokenize', choices=['ngram', 'word', 'mixed'], default='word')
    p.add_argument('--dump-keywords', type=str, default='')
    return p.parse_args()


def normalize_word(word):
    if num_re.match(word):
        return '_NUM_'
    word = punct_re.sub('', word)
    if args.lower:
        word = word.lower()
    return word.strip()


def tokenize(doc, args):
    if args.tokenize == 'word' or args.tokenize == 'mixed':
        for word in word_tokenize(doc[0].lower() + doc[1:]):
            tr = normalize_word(word)
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
            trimmed = normalize_word(word)
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
            trimmed = normalize_word(word)
            if trimmed:
                tf[trimmed] = 1
    return tf


def tf_rawfreq(doc_stream, args):
    tf = defaultdict(int)
    for line in doc_stream:
        for word in tokenize(line.decode('utf8').strip(), args):
            trimmed = normalize_word(word)
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
            tfidf_top[doc][term] = score
            n -= 1
    return tfidf_top


def train(args):
    dir_ = args.train
    doc_tfs = {}
    for fn in sorted(os.listdir(dir_)):
        stderr.write(fn + '\n')
        with open(os.path.join(dir_, fn)) as f:
            doc_tfs[fn] = tf(f, args)
    idf_d = idf(doc_tfs, args.idf)
    return tfidf(doc_tfs, idf_d, top=args.topn), idf_d


def test(tfidf, idf, args):
    langs = sorted(os.listdir(args.test))
    results = defaultdict(lambda: defaultdict(int))
    for fn in sorted(os.listdir(args.test)):
        with open(os.path.join(args.test, fn)) as f:
            for doc in f:
                guess, allguess = classify_text(doc.decode('utf8'), tfidf, idf)
                prob_str = ' '.join(str(allguess.get(l, 0)) for l in langs)
                print('{0} {1} {2} {3} {4}'.format(langs.index(fn), langs.index(guess[0]), prob_str, len(doc.decode('utf8').split()), len(allguess)))
                results[fn][guess[0]] += 1
    return results


def classify_text(doc, tfidf, idf):
    hits = defaultdict(int)
    test_terms = get_test_terms(doc, idf)
    for term, weight in test_terms.iteritems():
        tr = normalize_word(term)
        if not tr.strip():
            continue
        for lang, terms in tfidf.iteritems():
            if term in terms:
                #print lang, term.encode('utf8'), terms[term], weight, terms[term] * weight
                in_test[lang].add(term)
                hits[lang] += terms[term] * weight
    if not hits:
        return ('UN', 0), {}
    return max(hits.iteritems(), key=lambda x: x[1]), hits


def get_test_terms(doc, idf):
    words = defaultdict(int)
    for w in doc.split():
        tr = normalize_word(w)
        if tr:
            words[tr] += 1
    weights = {}
    max_f = max(words.iteritems(), key=lambda x: x[1])[1]
    for word, f in words.iteritems():
        if not word in idf:
            continue
        if args.qf == 'smooth':
            weights[word] = (0.5 + 0.5 * float(f) / max_f) * idf[word]
        elif args.qf == 'lognorm':
            weights[word] = math.log(1 + idf[word])
        elif args.qf == 'smoothnorm':
            weights[word] = (1 + float(f)) * idf[word]
        elif args.qf == 'raw':
            weights[word] = f
    return weights


def print_stats(t):
    good = 0
    N = 0
    for tr, guesses in t.iteritems():
        good += guesses.get(tr, 0)
        N += sum(guesses.itervalues())
    t['acc'] = float(good) / N
    print json.dumps(t)
    return t


in_test = defaultdict(set)


def main():
    tfidf, idf = train(args)
    stderr.write('Trained\n')
    if args.dump_keywords:
        with open(args.dump_keywords, 'w') as f:
            for lang, terms in sorted(tfidf.iteritems()):
                for term, score in sorted(terms.iteritems(), key=lambda x: -x[1]):
                    f.write(u'{0}\t{1}\t{2}\n'.format(lang, term, score).encode('utf8'))
    t = test(tfidf, idf, args)
    for lang, s in sorted(in_test.iteritems()):
        print lang, len(s)
    stats = print_stats(t)
    with open('res/global_stats', 'a+') as f:
        if args.comment:
            f.write('{0}\t{1}\t{2}\n'.format('\t'.join(map(str, (args.train, args.test, args.topn, args.tokenize, args.tf, args.idf, args.qf, args.rare, args.lower))), json.dumps(stats)), args.comment)
        else:
            f.write('{0}\t{1}\n'.format('\t'.join(map(str, (args.train, args.test, args.topn, args.tokenize, args.tf, args.idf, args.qf, args.rare, args.lower))), json.dumps(stats)))

if __name__ == '__main__':
    args = parse_args()
    main()
