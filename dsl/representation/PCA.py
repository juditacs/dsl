from sklearn.decomposition import PCA
import numpy


class PcaEncoder(object):

    def __init__(self, dimension):
        self.dimension = dimension
        self.model = PCA(n_components=self.dimension, whiten=True)

    def train(self, data):
        self.data = data
        self.model.fit(data)
        #print(self.model.explained_variance_ratio_)

    def encode(self, vector):
        return self.model.transform(vector.transpose()).transpose()

    @property
    def repr_model(self):
        return self.model.transform(self.data)


if __name__ == '__main__':
    N = 18000
    obs = numpy.matrix(numpy.random.random((10,N))).transpose()
    encoder = PcaEncoder(dimension=2)
    #encoder.train(numpy.array([[1,1,1,0,0,0],[1,0,1,0,0,0],[1,1,1,0,0,0],[0,0,1,1,1,0], [0,0,1,1,0,0],[0,0,1,1,1,0]]))
    #encoded=encoder.encode(numpy.array([1,1,1,1,1,1]))
    encoder.train(obs)
    tobs = obs[1, :].transpose()
    encoded=encoder.encode(tobs)
    print(encoded)
