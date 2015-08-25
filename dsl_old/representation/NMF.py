import numpy as np
from sklearn.decomposition import ProjectedGradientNMF


class NmfEncoder(object):

    def __init__(self, latentDimension):
        self.latentDimension = latentDimension
        self.visibleDimension = None
        self.model = None
        self.data = None

    def train(self, matrix):
        if self.model is None:
            self.data = matrix
            self.visibleDimension = matrix.shape[1]
            self.model = ProjectedGradientNMF(n_components=self.latentDimension, init='random', random_state=0)
        self.model.fit(matrix)
        
    def encode(self, vector):
        return self.model.transform(vector)
        
    @property
    def repr_model(self):
        return self.model.transform(self.data)
    
if __name__ == '__main__':
    with open("../../dat/testMatrix.small.txt") as f:
        obs = np.loadtxt(f, np.int32)
    encoder = NmfEncoder(latentDimension=10)
    encoder.train(obs)
    print(encoder.encode(obs[1, :]))
