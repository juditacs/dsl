from sklearn.naive_bayes import GaussianNB
from sklearn import svm

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
        self.clf = svm.SVC(*args, **kwargs)

    def train(self, model, target):
        self.clf.fit(model, target)
    
    def classify_vector(self, vector):
        return self.clf.predict(vector)

