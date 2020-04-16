from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
from sklearn import datasets
from nolearn.dbn import DBN
import numpy as np
#import cv2

class DbnEncoder(object):

    def __init__(self, latentDimensions):
        self.latentDimensions = latentDimensions
        self.visibleDimension = None
        self.model = None
        self.data = None
        
    def train(self, matrix):
        if self.model is None:
            self.data = matrix
            self.visibleDimension = matrix.shape[1]
            self.model = DBN(self.latentDimensions.prepend(self.visibleDimension), learn_rates = 0.3, learn_rate_decays = 0.9, epochs = 10, verbose = 1) 
        self.model.fit(matrix)

    def encode(self, vector):
        return self.model.predict(vector)

    @property
    def repr_model(self):
        return self.model.predict(self.data)
    
if __name__ == '__main__':
    with open("../../dat/testMatrix.small.txt") as f:
        obs = np.loadtxt(f, np.int32)
    encoder = DbnEncoder(latentDimensions=[100, 10])
    encoder.train(obs)
    print(encoder.encode(obs[1, :]))
