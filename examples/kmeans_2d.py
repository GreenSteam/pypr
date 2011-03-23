from matplotlib.pylab import *
from pypr.clustering.kmeans import *
from pypr.clustering.rnd_gauss_clusters import *

# Make three clusters:
centroids = [ array([1,1]), array([3,3]), array([3,0]) ]
ccov = [ array([[1,0],[0,1]]), array([[1,0],[0,1]]), \
          array([[0.2, 0],[0, 0.2]]) ]
X = rnd_gauss_clusters(centroids, ccov, 1000)

figure(figsize=(10,5))
subplot(121)
title('Original unclustered data')
plot(X[:,0], X[:,1], '.')
xlabel('$x_1$'); ylabel('$x_2$')

subplot(122)
title('Clustered data')
m, cc = kmeans(X, 3)
plot(X[m==0, 0], X[m==0, 1], 'r.')
plot(X[m==1, 0], X[m==1, 1], 'b.')
plot(X[m==2, 0], X[m==2, 1], 'g.')
xlabel('$x_1$'); ylabel('$x_2$')
