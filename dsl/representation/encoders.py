from PCA import PcaEncoder


class BaseEncoder(object):

    @staticmethod
    def lookup_encoder(enc_name, *args, **kwargs):
        if isinstance(enc_name, str):
            if enc_name.lower() == 'pca':
                return PcaEncoder(*args, **kwargs)
        raise ValueError('Unknown encoder: {}'.format(enc_name))
