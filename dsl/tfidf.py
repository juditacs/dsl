import math
from collections import defaultdict


class Tfidf(object):

    def __init__(self, name, tf, idf, qf, topn, rare, lower):
        # this is going to be the prefix for features
        self.name = name
        self.tf_weight = tf
        self.idf_weight = idf
        self.qf_weight = qf
        self.topn = topn
        self.rare = rare
        self.lower = lower
        self.tokens_by_lang = defaultdict(lambda: defaultdict(int))
        self.global_freq = defaultdict(int)

    def train(self, sentences):
        self.count_tokens(sentences)
        self.compute_tf()
        self.compute_idf()
        self.compute_tfidf()
        delattr(self, 'global_freq')
        delattr(self, 'tokens_by_lang')
        delattr(self, 'tf')
        delattr(self, 'idf')

    def compute_tf(self):
        if self.tf_weight == 'raw':
            self.compute_raw_tf()
        elif self.tf_weight == 'binary':
            self.compute_binary_tf()
        elif self.tf_weight == 'lognorm':
            self.compute_lognorm_tf()
        else:
            raise ValueError('Unknown tf type: {}'.format(self.tf))

    def compute_raw_tf(self):
        self.tf = defaultdict(dict)
        for label, term, cnt in self.get_frequent_terms():
            self.tf[label][term] = cnt

    def compute_binary_tf(self):
        self.tf = defaultdict(dict)
        for label, term, cnt in self.get_frequent_terms():
            self.tf[label][term] = 1

    def compute_lognorm_tf(self):
        self.tf = defaultdict(dict)
        for label, term, cnt in self.get_frequent_terms():
            self.tf[label][term] = math.log(1 + cnt)

    def get_frequent_terms(self):
        for label, terms in self.tokens_by_lang.iteritems():
            for term, cnt in terms.iteritems():
                if cnt >= self.rare:
                    yield label, term, cnt

    def compute_idf(self):
        self.df_cnt = defaultdict(int)
        for term, cnt in self.global_freq.iteritems():
            if cnt < self.rare:
                continue
            for doc_terms in self.tf.itervalues():
                if term in doc_terms:
                    self.df_cnt[term] += 1
        if self.idf_weight == 'invfreq':
            self.compute_invfreq_idf()
        elif self.idf_weight == 'smooth':
            self.compute_smooth_idf()
        else:
            raise ValueError('Unknown idf weighting: {}'.format(self.idf_weight))

    @property
    def N(self):
        return len(self.tokens_by_lang)

    def compute_invfreq_idf(self):
        self.idf = {}
        for term, cnt in self.df_cnt.iteritems():
            self.idf[term] = math.log(self.N / float(cnt))

    def compute_smooth_idf(self):
        self.idf = {}
        for term, cnt in self.df_cnt.iteritems():
            self.idf[term] = math.log(1 + self.N / float(cnt))

    def compute_tfidf(self):
        tfidf = defaultdict(dict)
        for label, terms in self.tf.iteritems():
            for term, tf_score in terms.iteritems():
                tfidf[label][term] = tf_score * self.idf[term]
        self.tfidf = {}
        for label, terms in tfidf.iteritems():
            self.tfidf[label] = {}
            for term, score in sorted(terms.iteritems(), key=lambda x: -x[1])[:self.topn]:
                self.tfidf[label][term] = score

    def count_tokens(self, sentences):
        for sentence in sentences:
            for token in sentence:
                self.global_freq[token] += 1
                self.tokens_by_lang[sentence.label][token] += 1

    def score_sentence(self, sentence):
        #TODO weight
        score = defaultdict(float)
        for lang, keywords in self.tfidf.iteritems():
            for token in sentence:
                if token in keywords:
                    score[lang] += keywords[token]
        for lang, sc in score.iteritems():
            sentence.features[u'{0}{1}'.format(self.name, lang)] = sc
