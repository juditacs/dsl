from encoders import BaseEncoder
from classifiers import BaseClassifier, LogisticRegressionWrapper


class Representation(object):

    def __init__(self, encoder, classifier, *args, **kwargs):
        enc_args = {}
        cl_args = {}
        klen = len(encoder) + 1
        clen = len(classifier) + 1
        for k, v in kwargs.iteritems():
            if k.startswith(encoder):
                enc_args[k[klen:]] = v
            elif k.startswith(classifier):
                cl_args[k[clen:]] = v
        self.encoder = BaseEncoder.lookup_encoder(encoder, **enc_args)
        self.classifier = BaseClassifier.lookup_classifier(classifier, **cl_args)

    def encode(self, matrix):
        self.encoder.train(matrix)

    def train_classifier(self, labels):
        self.classifier.train(self.encoder.repr_model, labels)

    def classify_vector(self, vector, with_probs=False):
        encoded = self.encoder.encode(vector)
        if isinstance(self.classifier, LogisticRegressionWrapper):
            return self.classifier.classify_vector(encoded, with_probs)
        else:
            return self.classifier.classify_vector(encoded)
