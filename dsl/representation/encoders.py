from PCA import PcaEncoder
from RBM import RbmEncoder


class BaseEncoder(object):

    @staticmethod
    def lookup_encoder(enc_name, *args, **kwargs):
        if isinstance(enc_name, str):
            if enc_name.lower() == 'pca':
                return PcaEncoder(*args, **kwargs)
            if enc_name.lower() == 'rbm':
                return RbmEncoder(*args, **kwargs)
        raise ValueError('Unknown encoder: {}'.format(enc_name))
