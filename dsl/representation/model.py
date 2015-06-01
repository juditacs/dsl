from encoders import BaseEncoder
from classifiers import BaseClassifier


class Representation(object):

    def __init__(self, encoder, classifier, *args, **kwargs):
        self.encoder = BaseEncoder.lookup_encoder(encoder, **kwargs)
        self.classifier = BaseClassifier.lookup_classifier(classifier)

    def encode(self, matrix):
        self.encoder.train(matrix)

    def train_classifier(self, labels):
        self.classifier.train(self.encoder.repr_model, labels)

    def classify_vector(self, vector):
        encoded = self.encoder.encode(vector)
        return self.classifier.classify_vector(encoded)
