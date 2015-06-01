from sklearn.naive_bayes import GaussianNB


class BaseClassifier(object):

    @staticmethod
    def lookup_classifier(cl_name):
        if isinstance(cl_name, str):
            if cl_name.lower() == 'naive_bayes':
                return NaiveBayesClassifier()
        raise ValueError('Unknown classifier: {}'.format(cl_name))


class NaiveBayesClassifier(BaseClassifier):

    def train(self, model, target):
        self.gnb = GaussianNB()
        self.gnb.fit(model, target)

    def classify_vector(self, vector):
        return self.gnb.predict(vector)
