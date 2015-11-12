from ssk_cython import *
import GPy
from GPy.util.caching import Cache_this

class WeightedDegree(GPy.kern.Kern):
    def __init__(self, input_dim, strings, d=4, variance=1., active_dims=None, name='weighted degree'):
        super(WeightedDegree, self).__init__(input_dim, active_dims, name)
        self.variance = GPy.core.parameterization.Param('variance', variance, GPy.core.parameterization.transformations.Logexp())
        self.link_parameters(self.variance)
        self.strings = strings
        self.string_kernel = cython_WD_K(self.strings.tolist(), self.strings.tolist(), d=d)

    def K(self, X, X2=None):
        if X2 is None:
            X2 = X
            
        Xind = np.asarray(X, dtype=int).flatten()
        X2ind = np.asarray(X2, dtype=int).flatten()
        K = self.string_kernel[Xind][:, X2ind]

        return self.variance * K

    def Kdiag(self, X):
        Xind = np.asarray(X, dtype=int).flatten()
        K = self.string_kernel[Xind][:, Xind]

        return self.variance * K.diagonal()

    def update_gradients_full(self, dL_dK, X, X2=None):
        if X2 is None:
            X2 = X

        Xind = np.asarray(X, dtype=int).flatten()
        X2ind = np.asarray(X2, dtype=int).flatten()
        K = self.string_kernel[Xind][:, X2ind]

        self.variance.gradient = np.einsum('ij,ij', dL_dK, K)

    def update_gradients_diag(self, dL_dKdiag, X):
        Xind = np.asarray(X, dtype=int).flatten()
        K = self.string_kernel[Xind][:, Xind]

        self.variance.gradient = np.einsum('i,i', dL_dKdiag, K)



if __name__ == '__main__':
    x1 = 'ATCGATCG'
    x2 = 'ATCGATCG'
    x3 = 'ATCGATCC'
    x4 = 'ATCGATAA'
    x5 = 'ANNNNNNN'
    X = np.arange(5)[:, None]
    y = np.random.randn(5, 1)
    kern = WeightedDegree(1, np.array([x1, x2, x3, x4, x5]))
    m = GPy.models.GPRegression(X, y, kernel=kern)
