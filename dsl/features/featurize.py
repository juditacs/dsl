import string
import re
import math
from os import listdir, path
from collections import defaultdict
from nltk.tokenize import word_tokenize
from scipy.sparse import dok_matrix
import numpy as np


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

    @staticmethod
    def count_ngrams(text, cnt=None, N=3, padding=False, include_shorter=True):
        if cnt is None:
            cnt = defaultdict(int)
        for i in xrange(0, len(text) - N + 1):
            ngram = cnt[text[i:i + N]]
            if include_shorter:
                for j in xrange(1, N):
                    cnt[text[i:i + j]] += 1
            cnt[text[i:i + N]] += 1
        if include_shorter:
            for i in xrange(len(text) - N + 1, len(text)):
                cnt[text[i:]] += 1
        return cnt

    @staticmethod
    def tokenize_stream_and_count(tokenizer, stream, N=3, padding=False):
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


class Featurizer(object):

    def __init__(self, tokenizer, N=3, extractor=CharacterNgram.tokenize_line_and_count):
        self.tokenizer = tokenizer
        self.N = N
        self.extractor = extractor
        self.featdict = FeatDict()
        self.labels = FeatList()

    def featurize_in_directory(self, basedir):
        self.docs = []
        for fn in sorted(filter(lambda x: not x.endswith('.swp'), listdir(basedir))):
            with open(path.join(basedir, fn)) as f:
                for l in f:
                    self.labels.append(fn)
                    newd = {}
                    for k, v in self.extractor(self.tokenizer, l, self.N, padding=True).iteritems():
                        featname = self.featdict[k]
                        if featname is None:
                            continue
                        newd[featname] = min((v, 0xffff))
                    self.docs.append(newd)
                    #self.docs.append({self.featdict[k]: min((v, 0xffff)) for k, v in self.extractor(self.tokenizer, l, self.N).iteritems()})
        return self.docs

    def get_correlations(self):
        label_mean, label_dev = self.get_label_mean_dev()
        self.label_mean = label_mean
        self.label_dev = label_dev
        feat_mean, feat_dev = self.get_feat_mean_dev()
        self.feat_mean = feat_mean
        self.feat_dev = feat_dev

    def get_feat_mean_dev(self):
        m = [0 for _ in xrange(len(self.featdict))]
        msq = [0 for _ in xrange(len(self.featdict))]
        for doc in self.docs:
            for feat, val in doc.iteritems():
                m[feat] += val
                msq[feat] += val ** 2
        N = float(len(self.docs))
        dev = []
        for i in xrange(len(m)):
            msq[i] /= N
            m[i] /= N
            dev.append(math.sqrt(msq[i] - m[i] ** 2))
        return m, dev

    def get_label_mean_dev(self):
        m = [0 for _ in range(len(self.labels))]
        for label in self.labels:
            m[label] += 1
        dev = []
        for i in xrange(len(m)):
            m[i] = m[i] / float(len(self.docs))
            dev.append(math.sqrt(m[i] - m[i] ** 2))
        return m, dev

    def label_feat_pearson(self):
        pearson = [[0 for i in range(len(self.labels))] for _ in range(len(self.featdict))]
        for doc_i, doc in enumerate(self.docs):
            doc_label = self.labels[doc_i]
            for i in xrange(len(self.featdict)):
                doc_val = doc.get(i, 0)
                pdoc = (doc_val - self.feat_mean[i])
                for lab_i in xrange(len(self.labels.features)):
                    lab_val = 1 if doc_label == lab_i else 0
                    pval = lab_val - self.label_mean[lab_i]
                    pearson[i][lab_i] += pdoc * pval
        for i in xrange(len(pearson)):
            for j in xrange(len(pearson[0])):
                if not self.feat_dev[i] == 0 and not self.label_dev[lab_i] == 0:
                    pearson[i][j] /= (self.feat_dev[i] * self.label_dev[j]) * len(self.docs)
                else:
                    pearson[i][j] = 0
        self.pearson = pearson

    def save_features(self, fn):
        with open(fn, 'w') as f:
            for doc in self.docs:
                out_l = []
                for ngram in self.topngrams:
                    out_l.append(doc.get(ngram, 0))
                f.write(' '.join(map(str, out_l)) + '\n')

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

    def to_dok_matrix(self):
        mtx = dok_matrix((len(self.docs), len(self.featdict)), dtype=np.int16)
        for i, doc in enumerate(self.docs):
            for j, val in doc.iteritems():
                mtx[(i, j)] = val
        return mtx

    def featurize_stream(stream, tokenizer, N=3, extractor=CharacterNgram.tokenize_line_and_count, feat_f='features'):
        documents = []
        for l in stream:
            doc_feats = extractor(tokenizer, l, N=N)
            documents.append(doc_feats)
        feat_mtx = dok_matrix((len(documents), len(all_features)), dtype=np.int16)
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
