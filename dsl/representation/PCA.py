from sklearn.decomposition import PCA
import numpy

class PcaEncoder(object):

    def __init__(self, dimension):
        self.dimension = dimension
        
    def train(self, data):
        self.data = data
        self.model = PCA(n_components=self.dimension,whiten=True)
        self.model.fit(data)
        #print(self.model.explained_variance_ratio_)
        
    def encode(self, vector):
        return self.model.transform(vector)

    @property
    def repr_model(self):
        return self.model.transform(self.data)

        
if __name__ == '__main__':
    encoder = PcaEncoder(dimension=2)
    encoder.train(numpy.array([[1,1,1,0,0,0],[1,0,1,0,0,0],[1,1,1,0,0,0],[0,0,1,1,1,0], [0,0,1,1,0,0],[0,0,1,1,1,0]]))
    encoded=encoder.encode(numpy.array([1,1,1,1,1,1]))
    print(encoded)
