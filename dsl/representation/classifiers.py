from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from nltk.classify.maxent import MaxentClassifier

class BaseClassifier(object):

    @staticmethod
    def lookup_classifier(cl_name, *args, **kwargs):
        if isinstance(cl_name, str):
            if cl_name.lower() == 'naive_bayes':
                return NaiveBayesClassifier()
            if cl_name.lower() == 'simple':
                return SimpleClassifier(*args, **kwargs)
            if cl_name.lower() == 'svm':
                return SVMClassifier(*args, **kwargs)
        raise ValueError('Unknown classifier: {}'.format(cl_name))


class SimpleClassifier(BaseClassifier):

    def __init__(self, field=0):
        self.field = field

    def train(self, model, target):
        pass

    def classify_vector(self, vector):
        return vector[self.field]


class NaiveBayesClassifier(BaseClassifier):

    def train(self, model, target):
        self.gnb = GaussianNB()
        self.gnb.fit(model, target)

    def classify_vector(self, vector):
        return self.gnb.predict(vector)


class SVMClassifier(BaseClassifier):

    def __init__(self, *args, **kwargs):
        ktype = kwargs['ktype'].lower()
        del kwargs['ktype']
        if ktype == 'svc':
            self.clf = svm.SVC(*args, **kwargs)
        elif ktype == 'linearsvc':
            self.clf = svm.LinearSVC(*args, **kwargs)
        elif ktype == 'nusvc':
            self.clf = svm.NuSVC(*args, **kwargs)
        else:
            raise ValueError('Unknown SVM type: {0}'.format(ktype))

    def train(self, model, target):
        self.clf.fit(model, target)
    
    def classify_vector(self, vector):
        return self.clf.predict(vector)

class MaxEntClassifier(BaseClassifier):

    def __init__(self, *args, **kwargs):
        self.maxent = MaxentClassifier(*args, **kwargs)

    def train(self, model, target):
        self.maxent.train(zip([{i:v for (i,v) in enumerate(l)} for l in model], target))

    def classify_vector(self, vector):
        return self.maxent.prob_classify({i:v for (i,v) in enumerate(l)})
