from encoders import BaseEncoder
from classifiers import BaseClassifier


class Representation(object):

    def __init__(self, encoder, classifier, *args, **kwargs):
        enc_args = {}
        cl_args = {}
        for k, v in kwargs.iteritems():
            if k.startswith(encoder):
                enc_args[k.lstrip(encoder + '_')] = v
            elif k.startswith(classifier):
                cl_args[k.lstrip(classifier + '_')] = v
        self.encoder = BaseEncoder.lookup_encoder(encoder, **enc_args)
        self.classifier = BaseClassifier.lookup_classifier(classifier, **cl_args)

    def encode(self, matrix):
        self.encoder.train(matrix)

    def train_classifier(self, labels):
        self.classifier.train(self.encoder.repr_model, labels)

    def classify_vector(self, vector):
        encoded = self.encoder.encode(vector)
        return self.classifier.classify_vector(encoded)
