import string
import re
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
    def count_ngrams(text, cnt=None, N=3, padding=False):
        if cnt is None:
            cnt = defaultdict(int)
        for i in xrange(0, len(text) - N + 1):
            cnt[text[i:i + N]] += 1
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

    def __getitem__(self, item):
        if not item in self.features:
            self.features[item] = self.max_i
            self.max_i += 1
        return self.features[item]

    def __len__(self):
        return len(self.features)

    def save(self, fn='features'):
        with open(fn, 'w') as f:
            for k, v in sorted(self.features):
                f.write(u'{0}\t{1}\n'.format(v, k))

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
        for fn in sorted(listdir(basedir)):
            with open(path.join(basedir, fn)) as f:
                self.labels.append(fn)
                docs = []
                for l in f:
                    docs.append({self.featdict[k]: min((v, 0xffff)) for k, v in self.extractor(self.tokenizer, l, self.N).iteritems()})
        return docs

    def to_dok_matrix(self, docs):
        mtx = dok_matrix((len(docs), len(self.featdict)), dtype=np.int16)
        for i, doc in enumerate(docs):
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
