# Drawing samples from a Gaussian Mixture Model
from numpy import *
from matplotlib.pylab import *
import pypr.clustering.gmm as gmm

mc = [0.4, 0.4, 0.2] # Mixing coefficients
centroids = [ array([0,0]), array([3,3]), array([0,4]) ]
ccov = [ array([[1,0.4],[0.4,1]]), diag((1,2)), diag((0.4,0.1)) ]
    # Covariance matrices

X = gmm.sample_gaussian_mixture(centroids, ccov, mc, samples=1000)
plot(X[:,0], X[:,1], '.')

for i in range(len(mc)):
    x1, x2 = gmm.gauss_ellipse_2d(centroids[i], ccov[i])
    plot(x1, x2, 'k', linewidth=2)
xlabel('$x_1$'); ylabel('$x_2$')
