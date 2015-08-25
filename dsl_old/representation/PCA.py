from sklearn.decomposition import PCA
import numpy as np


class PcaEncoder(object):

    def __init__(self, latentDimension):
        self.latentDimension = latentDimension
        self.model = PCA(n_components=self.latentDimension, whiten=True)

    def train(self, data):
        self.data = data
        self.model.fit(data)
        #print(self.model.explained_variance_ratio_)

    def encode(self, vector):
        return self.model.transform(vector)

    @property
    def repr_model(self):
        return self.model.transform(self.data)


if __name__ == '__main__':
    with open("../../dat/testMatrix.small.txt") as f:
        obs = np.loadtxt(f, np.int32)
    encoder = PcaEncoder(latentDimension=50)
    encoder.train(obs)
    print(encoder.encode(obs[1, :]))
