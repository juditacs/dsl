# -*- coding: utf-8 -*-
import re
import os
import string

from tfidf import Tfidf


class Sentence(object):

    def __init__(self, orig, label):
        if isinstance(orig, unicode):
            self.orig = orig
        else:
            self.orig = orig.decode('utf8')
        self.tokens = Tokenizer.trim_and_tokenize(orig)
        self.label = label
        self.features = {}
        self.__prefix = ''

    def __iter__(self):
        for t in self.tokens:
            yield t

    def __unicode__(self):
        return self.orig

    def __str__(self):
        return self.orig.encode('utf8')

    def __repr__(self):
        return str(self)

    def __len__(self):
        return len(self.tokens)

    def set_prefix(self, pre):
        self.__prefix = pre

    def extract_ngram_features(self, n, padding, lower=True, shorter_too=False):
        """ extract ngrams from tokens """
        if shorter_too:
            for N in xrange(1, n):
                self.__extract_ngram_features(N, padding=padding, lower=lower)
        self.__extract_ngram_features(N, padding=padding, lower=lower)

    def __extract_ngram_features(self, n, padding, lower=True):
        if padding:
            sp = ' ' * (n - 1)
        else:
            sp = ' '
        sent = sp.join(self.tokens)
        if lower:
            sent = sent.lower()
        if padding:
            sent = u'{0}{1}{0}'.format(sp, sent)
        for i in xrange(0, len(sent) - n + 1):
            ngram = u'{0}{1}'.format(self.__prefix, sent[i:i + n])
            if not ngram in self.features:
                self.features[ngram] = 0
            self.features[ngram] += 1


class Tokenizer(object):

    ws_re = re.compile(r'\s+', re.UNICODE)
    punct_str = string.punctuation + '…«­–´·’»”“„'
    punct_re = re.compile(r'[{0}]'.format(re.escape(punct_str)), re.UNICODE)
    digit_re = re.compile(r'[0-9,.-]+', re.UNICODE)
    trim_chain = [
        lambda x: Tokenizer.punct_re.sub(' ', x),
        lambda x: Tokenizer.ws_re.sub(' ', x),
        lambda x: x.strip(),
        lambda x: Tokenizer.digit_re.sub('0', x),
    ]

    @staticmethod
    def trim_and_tokenize(text):
        return Tokenizer.tokenize(Tokenizer.trim(text.decode('utf8')))

    @staticmethod
    def trim(text):
        trim = Tokenizer.trim_chain[0](text)
        for t in Tokenizer.trim_chain[1:]:
            trim = t(trim)
        return trim

    @staticmethod
    def tokenize(text):
        return text.split()


class SentenceCollection(object):

    def __init__(self, config):
        self.sentences = []
        self.config = config
        self.tfidf = None

    def __iter__(self):
        for sen in self.sentences:
            yield sen

    def __len__(self):
        return len(self.sentences)

    def read_sentences(self, train_dir, labeled=False):
        for fn in os.listdir(train_dir):
            with open(os.path.join(train_dir, fn)) as f:
                for l in f:
                    self.sentences.append(Sentence(l, label=fn))

    def featurize(self):
        for name in self.config.sections():
            if not name.startswith('featurize_'):
                continue
            fname = name[10:]
            if fname.startswith('ngram'):
                self.extract_ngram_features(name)
            elif fname.startswith('tfidf'):
                if self.tfidf is None:
                    self.train_tfidf(name)
                self.featurize_tfidf()

    def train_tfidf(self, name):
        tf = self.config.get(name, 'tf')
        idf = self.config.get(name, 'idf')
        qf = self.config.get(name, 'qf')
        topn = self.config.getint(name, 'topn')
        rare = self.config.getint(name, 'rare')
        lower = self.config.getboolean(name, 'lower')
        self.tfidf = Tfidf(name=name[10:], tf=tf, idf=idf, qf=qf, topn=topn, rare=rare, lower=lower)
        self.tfidf.train(self)

    def extract_ngram_features(self, name):
        fname = name[10:]
        n = self.config.getint(name, 'n')
        padding = self.config.getboolean(name, 'padding')
        lower = self.config.getboolean(name, 'lower')
        shorter_too = self.config.getboolean(name, 'extract_shorter_too')
        for i, sentence in enumerate(self.sentences):
            sentence.set_prefix(fname)
            sentence.extract_ngram_features(n, lower=lower, padding=padding, shorter_too=shorter_too)

    def featurize_tfidf(self):
        for sentence in self.sentences:
            self.tfidf.score_sentence(sentence)
