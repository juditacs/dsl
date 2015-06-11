import string
import re
import math
from os import listdir, path
from sys import stderr
from collections import defaultdict
from nltk.tokenize import word_tokenize
from scipy.sparse import dok_matrix
import numpy as np
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
    def tokenize_line_and_count(tokenizer, text, N=3, padding=False):
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
                f.write(u'{0}\t{1}\n'.format(v, k).encode('utf8'))

    def load(self, fn='features'):
        with open(fn) as f:
            for l in f:
                fs = l.decode('utf8').rstrip().split('\t')
                self.features[fs[1]] = int(fs[0])
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
                    for k, v in self.extractor(self.tokenizer, l, self.N, padding=False).iteritems():
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
        self.pearson = pearson_mtx(np.array(self.to_dok_matrix().todense()), self.labels.to_dense_mtx())
        if strategy == 'simple':
            topfeats = sorted(((i, self.pearson[i, :]) for i in xrange(self.pearson.shape[0])), key=lambda x: np.max(x[1]), reverse=True)[:topn]
        elif strategy == 'diff':
            topfeats = sorted(((i, self.pearson[i, :]) for i in xrange(self.pearson.shape[0])), key=lambda x: abs(x[1][0] - x[1][1]), reverse=True)[:topn]

        self.topngrams = set(i[0] for i in topfeats)

    def save_features(self, stream):
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

    def to_dok_matrix(self):
        mtx = dok_matrix((len(self.docs), len(self.featdict)), dtype=np.int64)
        for i, doc in enumerate(self.docs):
            for j, val in doc.iteritems():
                mtx[(i, j)] = val
        return mtx

    def featurize_stream(stream, tokenizer, N=3, extractor=CharacterNgram.tokenize_line_and_count, feat_f='features'):
        documents = []
        for l in stream:
            doc_feats = extractor(tokenizer, l, N=N)
            documents.append(doc_feats)
        feat_mtx = dok_matrix((len(documents), len(all_features)), dtype=np.int65)
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
