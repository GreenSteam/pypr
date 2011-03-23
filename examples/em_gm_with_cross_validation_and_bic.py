# Drawing samples from a Gaussian Mixture Model
from numpy import *
from matplotlib.pylab import *
import pypr.clustering.gmm as gmm
import pypr.stattest as stattest

seed(10)
mc = [0.4, 0.4, 0.2] # Mixing coefficients
centroids = [ array([0,0]), array([3,3]), array([0,4]) ]
ccov = [ array([[1,0.4],[0.4,1]]), diag((1,2)), diag((0.4,0.1)) ]
    # Covariance matrices

T = gmm.sample_gaussian_mixture(centroids, ccov, mc, samples=500)
V = gmm.sample_gaussian_mixture(centroids, ccov, mc, samples=500)
plot(T[:,0], T[:,1], '.')

# Expectation-Maximization of Mixture of Gaussians
Krange = range(1, 20 + 1);
runs = 1
meanLogL_train = np.zeros((len(Krange), runs))
meanLogL_valid = np.zeros((len(Krange), runs))
for K in Krange:
    print "Clustering for K = ", K; sys.stdout.flush()
    for r in range(runs):
        cen_lst, cov_lst, p_k, logL = gmm.em_gm(T, K = K, iter = 100)
        meanLogL_train[K-1, r] = logL
        meanLogL_valid[K-1, r] = gmm.gm_log_likelihood(V, cen_lst, cov_lst, p_k)

fig1 = figure()
subplot(1, 2, 1)
for r in range(runs):
    plot(Krange, meanLogL_train[:, r], 'g:', label='Training')
    plot(Krange, meanLogL_valid[:, r], 'b-', label='Validation')
legend(loc='lower right')
xlabel('Number of clusters')
ylabel('log likelihood')

bic = np.zeros(len(Krange))
# We should train with ALL data here
X = np.concatenate((T,V), axis = 0)
meanLogL_full = np.zeros(len(Krange))
for i, K in enumerate(Krange):
    print "Clustering for K = ", K; sys.stdout.flush()
    for r in range(runs):
        cen_lst, cov_lst, p_k, logL = gmm.em_gm(X, K = K, iter = 100)
        meanLogL_full[i] += logL
meanLogL_full /= runs
for i, K in enumerate(Krange):
    D = 2
    M = (K-1) + K*(D+0.5*D*(D+1))
    N = X.shape[0]
    bic[i] = stattest.bic(meanLogL_full[i], M, N)
subplot(1, 2, 2)
plot(Krange, bic)
xlabel('Number of clusters')
ylabel('BIC score')
