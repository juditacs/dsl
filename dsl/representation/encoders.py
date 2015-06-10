from PCA import PcaEncoder
from RBM import RbmEncoder
from IPCA import IpcaEncoder


class BaseEncoder(object):

    @staticmethod
    def lookup_encoder(enc_name, *args, **kwargs):
        if isinstance(enc_name, str):
            if enc_name.lower() == 'pca':
                return PcaEncoder(*args, **kwargs)
            if enc_name.lower() == 'dummy':
                return DummyEncoder()
            if enc_name.lower() == 'rbm':
                return RbmEncoder(*args, **kwargs)
            if enc_name.lower() == 'ipca':
                return IpcaEncoder(*args, **kwargs)
        raise ValueError('Unknown encoder: {}'.format(enc_name))

class DummyEncoder(BaseEncoder):

    def train(self, data):
        self.data = data
        self.model = data

    def encode(self, vector):
        return vector

    @property
    def repr_model(self):
        return self.model
