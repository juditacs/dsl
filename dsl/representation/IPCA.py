import numpy as np

_ZERO_THRESHOLD = 1e-9  # Everything below this is zero

__author__ = "Micha Kalfon"

class IpcaEncoder(object):

    def __init__(self, pcaDimension):
        self.pcaDimension = pcaDimension
        self.sampleDimension = None
        self.model = None
        
    def train(self, vector):
        if self.model is None:
            # The size of the vector
            self.sampleDimension = vector.shape[0]
            self.model = IPCA(self.sampleDimension, self.pcaDimension)
        self.model.update(vector)
        
    def encode(self, vector):
        if self.model is None:
            raise "The model is not trained, yet"
        # Select the first n PCA components and multiple the vector with it.
        return self.model.components.transpose() * vector

    def getModel(self):
        return self.model.components

class IPCA(object):
    """Incremental PCA calculation object.

    General Parameters:
        m - Number of variables per observation
        n - Number of observations
        p - Dimension to which the data should be reduced
    """

    def __init__(self, m, p):
        """Creates an incremental PCA object for m-dimensional observations
        in order to reduce them to a p-dimensional subspace.

        @param m: Number of variables per observation.
        @param p: Number of principle components.

        @return: An IPCA object.
        """
        self._m = float(m)
        self._n = 0.0
        self._p = float(p)
        self._mean = np.matrix(np.zeros((m , 1), dtype=np.float64))
        self._covariance = np.matrix(np.zeros((m, m), dtype=np.float64))
        self._eigenvectors = np.matrix(np.zeros((m, p), dtype=np.float64))
        self._eigenvalues = np.matrix(np.zeros((1, p), dtype=np.float64))

    def update(self, x):
        """Updates with a new observation vector x.

        @param x: Next observation as a column vector (m x 1).
        """
        m = self._m
        n = self._n
        p = self._p
        mean = self._mean
        C = self._covariance
        U = self._eigenvectors
        E = self._eigenvalues

        if type(x) is not np.matrix or x.shape != (m, 1):
            raise TypeError('Input is not a matrix (%d, 1)' % int(m))

        # Update covariance matrix and mean vector and centralize input around
        # new mean
        oldmean = mean
        mean = (n * mean + x) / (n + 1.0)
        C = (n * C + x * x.T + n * oldmean * oldmean.T - (n + 1) * mean * mean.T) / (n + 1.0)
        x -= mean

        # Project new input on current p-dimensional subspace and calculate
        # the normalized residual vector
        g = U.T * x
        r = x - (U * g)
        r = (r / np.linalg.norm(r)) if not _is_zero(r) else np.zeros_like(r)

        # Extend the transformation matrix with the residual vector and find
        # the rotation matrix by solving the eigenproblem DR=RE
        U = np.concatenate((U, r), 1)
        D = U.T * C * U
        (E, R) = np.linalg.eigh(D)

        # Sort eigenvalues and eigenvectors from largest to smallest to get the
        # rotation matrix R
        sorter = list(reversed(E.argsort(0)))
        E = E[sorter]
        R = R[:, sorter]

        # Apply the rotation matrix
        U = U * R       

        # Select only p largest eigenvectors and values and update state
        self._n += 1.0
        self._mean = mean
        self._covariance = C
        self._eigenvectors = U[:, 0:p]
        self._eigenvalues = E[0:p]

    @property
    def components(self):
        """Returns a matrix with the current principal components as columns.
        """
        return self._eigenvectors

    @property
    def variances(self):
        """Returns a list with the appropriate variance along each principal
        component.
        """
        return self._eigenvalues

def _is_zero(x):
    """Return a boolean indicating whether the given vector is a zero vector up
    to a threshold.
    """
    return np.fabs(x).min() < _ZERO_THRESHOLD


if __name__ == '__main__':
    import sys

    def pca_svd(X):
        X = X - X.mean(0).repeat(X.shape[0], 0)
        [_, _, V] = np.linalg.svd(X)
        return V

    N = 18000
    #obs = np.matrix([np.random.normal(size=10) for _ in xrange(N)])
    obs = np.matrix(np.random.random((10,N))).transpose()

    ipcaEncoder = IpcaEncoder(2)
    ipca = IPCA(obs.shape[1], 2)
    ipca = IPCA(10, 2)
    for i in xrange(obs.shape[0]):
        x = obs[i, :].transpose()
        ipca.update(x)
        ipcaEncoder.train(x)

    U = pca_svd(obs)[0:2].transpose()
    V = ipca.components
    W = ipcaEncoder.getModel()

    # print V
    # print U
    # print U-V
    # print U-W
    # print V-W
    tobs = obs[1, :].transpose()
    ru = U.transpose() * tobs
    rv = V.transpose() * tobs
    rw = W.transpose() * tobs
    print ru-rv
    print ru-rw
    print rv-rw
    # print obs[1,:] * U
    # print 
    # print pcaEncoder.encode(obs[1,:].transpose())
    # print U - pcaEncoder.getModel()
