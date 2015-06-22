import string
import re
import math
import logging
from os import listdir, path
from sys import stderr
from collections import defaultdict
from nltk.tokenize import word_tokenize
from scipy.sparse import dok_matrix, lil_matrix
from scipy.io import mmread, mmwrite
import numpy as np
from sklearn.preprocessing import scale
from dsl.utils.pearson import pearson_mtx


class Tokenizer(object):

    def __init__(self, lower=True, filter_punct=True, ws_norm=True, strip=True, replace_digits=True):
        self.to_lower = lower
        self.filter_punct = filter_punct
        self.ws_norm = ws_norm
        self.to_strip = strip
        self.replace_digits = replace_digits
        self.setup_trim_chain()

    def setup_trim_chain(self):
        self.trim_chain = []
        if self.to_lower:
            self.trim_chain.append(lambda x: x.lower())
        if self.filter_punct:
            punct_re = re.compile(r'[{0}]'.format(re.escape(string.punctuation)), re.UNICODE)
            self.trim_chain.append(lambda x: punct_re.sub(' ', x))
        if self.ws_norm:
            ws_re = re.compile(r'\s+', re.UNICODE)
            self.trim_chain.append(lambda x: ws_re.sub(' ', x))
        if self.to_strip:
            self.trim_chain.append(lambda x: x.strip())
        if self.replace_digits:
            digit_re = re.compile(r'[0-9]+', re.UNICODE)
            self.trim_chain.append(lambda x: digit_re.sub('0', x))

    def trim_text(self, text):
        for method in self.trim_chain:
            text = method(text)
        return text

    def tokenize_stream(self, stream):
        for l in stream:
            trimmed = self.trim_text(l.decode('utf8'))
            yield word_tokenize(trimmed)

    def tokenize_line(self, line):
        trimmed = self.trim_text(line.decode('utf8'))
        return word_tokenize(trimmed)


class WordNgram(object):

    def __init__(self, tokenizer, N=3, padding=False):
        self.ngrams = defaultdict(int)
        self.tokenizer = tokenizer
        self.N = N
        if padding:
            self.padding = ['*PAD*'] * (N - 1)
        else:
            self.paddin = False

    def count_in_file(self, basedir, N, padding=False):
        with open(path.join(basedir, fn)) as f:
            self.count_words(f)

    def count_words_in_stream(self, stream):
        for sentence in self.tokenizer.tokenize_stream(stream):
            self.count_in_sentence(sentence)

    def count_in_sentence(self, sentence, cnt=None):
        if self.padding:
            sentence = self.padding + sentence + self.padding
        for i in xrange(0, len(sentence) - self.N + 1):
            self.ngrams[tuple(sentence[i:i + self.N])] += 1

    def get_topn_words(self, n):
        topwords = set()
        for k, v in sorted(self.ngrams.iteritems(), key=lambda x: -x[1])[:n]:
            topwords |= set(k)
        return topwords


class CharacterNgram(object):

    def __init__(self, tokenizer):
        self.ngrams = defaultdict(int)
        self.tokenizer = tokenizer

    def count_in_directory(self, basedir, N, padding=False):
        for fn in sorted(filter(lambda x: not x.endswith('.swp'), listdir(basedir))):
            with open(path.join(basedir, fn)) as f:
                CharacterNgram.tokenize_stream_and_count(self.tokenizer, f, N=N, padding=padding, cnt=self.ngrams)

    def get_frequent(self, threshold):
        return {k: v for k, v in self.ngrams.iteritems() if v >= threshold}

    @staticmethod
    def count_ngrams(text, cnt=None, N=3, padding=False, include_shorter=True):
        if cnt is None:
            cnt = defaultdict(int)
        for i in xrange(0, len(text) - N + 1):
            if include_shorter:
                for j in xrange(1, N):
                    cnt[text[i:i + j]] += 1
            cnt[text[i:i + N]] += 1
        if include_shorter:
            for i in xrange(len(text) - N + 1, len(text)):
                cnt[text[i:]] += 1
        return cnt

    @staticmethod
    def tokenize_stream_and_count(tokenizer, stream, N=3, padding=False, cnt=None):
        if cnt is None:
            cnt = defaultdict(int)
        for sentence in tokenizer.tokenize_stream(stream):
            if padding:
                text = (' ' * (N - 1)).join(sentence)
            else:
                text = ' '.join(sentence)
            CharacterNgram.count_ngrams(text, cnt, N, padding)
        return cnt

    @staticmethod
    def tokenize_line_and_count(text, tokenizer, N=3, padding=False):
        cnt = defaultdict(int)
        words = tokenizer.tokenize_line(text)
        if padding:
            text = (' ' * (N - 1)).join(words)
        else:
            text = ' '.join(words)
        CharacterNgram.count_ngrams(text, cnt, N, padding)
        return cnt

    @staticmethod
    def filter_rare(cnt, rare=5):
        filt_cnt = defaultdict(int)
        for ngram, c in cnt.iteritems():
            if c > rare:
                filt_cnt[ngram] = c
        return filt_cnt


class FeatDict(object):

    def __init__(self):
        self.features = {}
        self.max_i = 0
        self._rev_d = None
        self.freeze = False

    def freeze_dict(self):
        self.freeze = True

    def __getitem__(self, item):
        if not item in self.features:
            if self.freeze:
                return None
            self.features[item] = self.max_i
            self.max_i += 1
        return self.features[item]

    def __len__(self):
        return len(self.features)

    def rev_lookup(self, item):
        if self._rev_d is None:
            self._rev_d = {v: k for k, v in self.features.iteritems()}
        return self._rev_d[item]

    def save(self, fn='features'):
        with open(fn, 'w') as f:
            for k, v in sorted(self.features.iteritems()):
                if isinstance(k, str):
                    f.write(u'{0}\t{1}\n'.format(v, k).encode('utf8'))
                elif isinstance(k, tuple):
                    f.write(u'{0}\t{1}\n'.format(v, '\t'.join(k)).encode('utf8'))

    def load(self, fn='features'):
        with open(fn) as f:
            for l in f:
                fs = l.decode('utf8').rstrip().split('\t')
                v = int(fs[0])
                if len(fs) > 2:
                    k = tuple(fs[1:])
                else:
                    k = fs[1]
                self.features[k] = v
            m = max(self.features.itervalues()) + 1
            self.max_i = self.max_i if m < self.max_i else m


class FeatList(object):

    def __init__(self):
        self.features = FeatDict()
        self.l = []
        self._rev_d = None

    def __len__(self):
        return len(self.features)

    def __getitem__(self, ind):
        return self.l[ind]

    def __iter__(self):
        for lab in self.l:
            yield lab

    def clear(self):
        self.l = []

    def load(self, fn):
        with open(fn) as f:
            for l in f:
                self.append(l.decode('utf8').strip())

    def save(self, fn, intlabels=False):
        with open(fn, 'w') as f:
            if intlabels:
                f.write('\n'.join(map(str, self.l)) + '\n')
            else:
                f.write('\n'.join(self.rev_lookup(l).encode('utf8') for l in self) + '\n')

    def load_featdict(self, fn):
        self.features.load(fn)

    def save_featdict(self, fn):
        self.features.save(fn)

    def rev_lookup(self, item):
        if self._rev_d is None:
            self._rev_d = {v: k for k, v in self.features.features.iteritems()}
        return self._rev_d[item]

    def append(self, item):
        self.l.append(self.features[item])

    def to_dense_mtx(self):
        mtx = np.zeros((len(self.l), len(self.features)))
        for i, label in enumerate(self.l):
            mtx[i, label] = 1
        return mtx


class Featurizer(object):

    def __init__(self, tokenizer, N=3, extractor=CharacterNgram.tokenize_line_and_count):
        self.tokenizer = tokenizer
        self.N = N
        self.extractor = extractor
        self.featdict = FeatDict()
        self.labels = FeatList()

    def featurize_in_directory(self, basedir, feature_filt=None):
        self.docs = []
        for fn in sorted(filter(lambda x: not x.endswith('.swp'), listdir(basedir))):
            with open(path.join(basedir, fn)) as f:
                for l in f:
                    self.labels.append(fn)
                    newd = {}
                    for k, v in self.extractor(l, self.tokenizer, self.N, padding=False).iteritems():
                        if feature_filt is not None and k not in feature_filt:
                            continue
                        featname = self.featdict[k]
                        if featname is None:
                            # if featdict is frozen
                            continue
                        newd[featname] = min((v, 0xffff))
                    self.docs.append(newd)
        return self.docs

    def choose_top_pearson(self, topn=500, strategy='simple'):
        self.pearson = pearson_mtx(np.array(self.dok_matrix.todense()), self.labels.to_dense_mtx())
        if strategy == 'simple':
            topfeats = sorted(((i, self.pearson[i, :]) for i in xrange(self.pearson.shape[0])), key=lambda x: np.max(x[1]), reverse=True)[:topn]
        elif strategy == 'diff':
            topfeats = sorted(((i, self.pearson[i, :]) for i in xrange(self.pearson.shape[0])), key=lambda x: abs(x[1][0] - x[1][1]), reverse=True)[:topn]

        self.topngrams = {i[0]: i[1] for i in topfeats}

    def save_top_corr_features(self):
        with open(self.paths['top_corr_features'], 'w') as f:
            #TODO
            f.write('\n'.join(u'{0}\t{1}\t{2}'.format(k, '\t'.join(map(str, v)), '\t'.join(self.featdict.rev_lookup(k))) for k, v in self.topngrams.iteritems()).encode('utf8'))

    def save_features(self, fn):
        if isinstance(fn, str):
            stream = open(fn, 'w')
        else:
            stream = fn
        if self.docs:
            self.save_by_doc(stream)
        else:
            self.save_by_matrix(stream)
        if isinstance(fn, str):
            stream.close()

    def to_filtered_matrix(self):
        mtx = lil_matrix((self.dok_matrix.shape[0], len(self.topngrams)), dtype=np.float64)
        j = 0
        cs = self.dok_matrix.tocsc()
        cs = scale(cs, with_mean=False)
        for i in xrange(cs.shape[1]):
            col = cs[:, i]
            if i in self.topngrams:
                mtx[:, j] = col
                j += 1
        self._dok_matrix = mtx.todok()

    def save_by_doc(self, stream):
        for doc in self.docs:
            out_l = []
            for ngram in self.topngrams:
                out_l.append(doc.get(ngram, 0))
            stream.write(' '.join(map(str, out_l)) + '\n')

    def filter_ngrams(self, docs, filt):
        feat_filt = []
        for doc in docs:
            feat_filt.append({})
            for feat, val in doc.iteritems():
                if feat in filt:
                    feat_filt[-1][feat] = val
        return feat_filt

    def filter_top_ngrams(self, topn):
        self.pearson_filt = sorted(enumerate(self.pearson), key=lambda x: max(map(abs, x[1])), reverse=True)[:topn]
        self.topngrams = set(i[0] for i in self.pearson_filt)
        feat_filt = []
        for doc in self.docs:
            feat_filt.append({})
            for feat, val in doc.iteritems():
                if feat in self.topngrams:
                    feat_filt[-1][feat] = val
        self.docs = feat_filt

    def write_pearson(self, fn, filter_top=False):
        with open(fn, 'w') as f:
            for i, p in enumerate(self.pearson):
                if filter_top and not i in self.topngrams:
                    continue
                f.write(u'{}\t{}\n'.format(self.featdict.rev_lookup(i), '\t'.join(map(str, p))).encode('utf8'))

    @property
    def dok_matrix(self):
        if not hasattr(self, '_dok_matrix'):
            self._dok_matrix = self.to_dok_matrix()
        return self._dok_matrix

    def load_matrix(self, fn):
        self._dok_matrix = mmread(fn)

    def save_matrix(self, fn):
        mmwrite(fn, self.dok_matrix)

    def save_as_dense_matrix(self, fn, matrix=None):
        if matrix is None:
            matrix = self.dok_matrix
        with open(fn, 'w') as f:
            for i, row in enumerate(matrix.todense()):
                f.write('{0} {1}\n'.format(self.labels[i], ' '.join(str(row[0, j]) for j in xrange(matrix.shape[1]))))

    def to_dok_matrix(self):
        mtx = dok_matrix((len(self.docs), len(self.featdict)), dtype=np.float64)
        for i, doc in enumerate(self.docs):
            for j, val in doc.iteritems():
                mtx[(i, j)] = val
        return mtx

    def featurize_stream(stream, tokenizer, N=3, extractor=CharacterNgram.tokenize_line_and_count, feat_f='features'):
        documents = []
        for l in stream:
            doc_feats = extractor(l, tokenizer, N=N)
            documents.append(doc_feats)
        feat_mtx = dok_matrix((len(documents), len(all_features)), dtype=np.float64)
        for i, doc in enumerate(documents):
            for feat, v in doc.iteritems():
                if not feat in feat_d:
                    feat_d[feat] = feat_i
                    feat_i += 1
                feat_mtx[(i, feat_d[feat])] = v
        with open(feat_f, 'w') as f:
            for feat, i in sorted(feat_d.iteritems(), key=lambda x: x[1]):
                f.write(u'{0}\t{1}\n'.format(feat, i).encode('utf8'))
        return feat_mtx


class BigramModel(Featurizer):

    def __init__(self, tokenizer, padding=False):
        self.docs = []
        self.tokenizer = tokenizer
        self.padding = padding
        self.featdict = FeatDict()
        self.labels = FeatList()

    def count_in_directory(self, basedir):
        bigrams = defaultdict(int)
        for fn in sorted(filter(lambda x: not x.endswith('.swp'), listdir(basedir))):
            with open(path.join(basedir, fn)) as f:
                for sentence in self.tokenizer.tokenize_stream(f):
                    if len(sentence) < 1:
                        continue
                    if self.padding:
                        bigrams[('^BEGIN', sentence[0])] += 1
                        bigrams[(sentence[-1], 'END$')] += 1
                    for i in xrange(len(sentence) - 1):
                        bigrams[(sentence[i], sentence[i + 1])] += 1
        self.bigrams = bigrams

    def get_frequent(self, threshold):
        freq = {}
        for bgr, cnt in self.bigrams.iteritems():
            if cnt >= threshold:
                freq[bgr] = cnt
        return set(freq.iterkeys())

    def featurize_directory(self, basedir, feature_filt=None):
        self.labels.clear()
        self.docs = []
        for fn in sorted(filter(lambda x: not x.endswith('.swp'), listdir(basedir))):
            with open(path.join(basedir, fn)) as f:
                for sentence in self.tokenizer.tokenize_stream(f):
                    self.labels.append(fn)
                    bigrams = defaultdict(int)
                    if len(sentence) == 0:
                        if self.padding:
                            bigrams[('BEGIN', 'END$')] += 1
                    else:
                        if self.padding:
                            bigrams[('^BEGIN', sentence[0])] += 1
                            bigrams[(sentence[-1], 'END$')] += 1
                        for i in xrange(len(sentence) - 1):
                            bigrams[(sentence[i], sentence[i + 1])] += 1
                    if feature_filt is not None:
                        b_filt = {}
                        for k, v in bigrams.iteritems():
                            if k in feature_filt:
                                b_filt[k] = v
                        self.docs.append(self.to_feats(b_filt))
                    else:
                        self.docs.append(self.to_feats(bigrams))

    def to_feats(self, bigrams):
        feats = {}
        for feat, cnt in bigrams.iteritems():
            featname = self.featdict[feat]
            if featname is not None:  # featdict is not frozen
                feats[featname] = cnt
        return feats

    def load_labels_from_dir(self, basedir):
        for fn in sorted(filter(lambda x: not x.endswith('.swp'), listdir(basedir))):
            with open(path.join(basedir, fn)) as f:
                for l in f:
                    self.labels.append(fn)

    def load_feature_filt(self, fn):
        with open(fn) as f:
            import cPickle
            self.feature_filt = cPickle.load(f)

    def save_feature_filt(self, fn):
        with open(fn, 'w') as f:
            import cPickle
            cPickle.dump(self.feature_filt, f)

    def load_or_build_matrix(self, raw_mtx_fn, basedir, threshold=0, filt=None, filt_fn='feature_filt'):
        if raw_mtx_fn and path.exists(raw_mtx_fn):
            self.load_matrix(raw_mtx_fn)
            self.load_labels_from_dir(basedir)
            self.load_feature_filt(filt_fn)
        else:
            if filt is None:
                freqcounter = BigramModel(self.tokenizer, padding=self.padding)
                freqcounter.count_in_directory(basedir)
                filt = freqcounter.get_frequent(threshold)
            self.feature_filt = filt
            self.featurize_directory(basedir, feature_filt=filt)
            self.save_feature_filt(filt_fn)
            if raw_mtx_fn:
                self.save_matrix(raw_mtx_fn)

    def load_frequent_filter(self, fn):
        with open(fn) as f:
            self.frequent_features = set(tuple(l.decode('utf8').strip().split()) for l in f)

    def save_frequent_filter(self, fn):
        with open(fn, 'w') as f:
            f.write('\n'.join(u'{} {}'.format(w[0], w[1]).encode('utf8') for w in self.frequent_features))

    def load_or_build_train(self, train_dir, threshold=0):
        if path.exists(self.paths['train_raw.mtx']):
            logging.debug('Path exists, loading matrix')
            self.load_matrix(self.paths['train_raw.mtx'])
            logging.debug('Loading featdict')
            self.featdict.load(self.paths['featdict'])
            logging.debug('Loading labeldict')
            self.labels.load_featdict(self.paths['labeldict'])
            logging.debug('Loading labels')
            self.labels.load(self.paths['train.labels'])
            logging.debug('Loading frequent features')
            self.load_frequent_filter(self.paths['frequent_features'])
        else:
            logging.debug('Path does not exist, creating matrix')
            freqcounter = BigramModel(self.tokenizer, padding=self.padding)
            freqcounter.count_in_directory(train_dir)
            self.frequent_features = freqcounter.get_frequent(threshold)
            logging.debug('Frequent features collected')
            self.save_frequent_filter(self.paths['frequent_features'])
            self.featurize_directory(train_dir, feature_filt=self.frequent_features)
            logging.debug('Directory [{}] featurized'.format(train_dir))
            self.save_matrix(self.paths['train_raw.mtx'])
            logging.debug('Matrix saved')
            self.featdict.save(self.paths['featdict'])
            self.labels.save_featdict(self.paths['labeldict'])
            self.labels.save(self.paths['train.labels'])
            self.labels.save(self.paths['train.labels.int'], intlabels=True)
            logging.debug('Saved all kinds of stuff')

    def load_or_build_test(self, test_dir):
        if path.exists(self.paths['test_raw.mtx']):
            logging.debug('Test matrix exists, loading')
            self.load_matrix(self.paths['test_raw.mtx'])
            self.labels.clear()
            self.labels.load(self.paths['test.labels'])
        else:
            logging.debug('Featurizeing test directory [{}]'.format(test_dir))
            self.featurize_directory(test_dir, feature_filt=self.frequent_features)
            self._dok_matrix = self.to_dok_matrix()
            self.labels.save(self.paths['test.labels.int'], intlabels=True)
            self.labels.save(self.paths['test.labels'])
            self.save_matrix(self.paths['test_raw.mtx'])
