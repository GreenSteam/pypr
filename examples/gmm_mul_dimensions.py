
"""This example illustrated the problem with Gaussian Mixture Models (GMM) when
the number of dimensions is large compared with the number of samples. You can
try to change the number of dimension D to 20, and the BIC score should become
to low. If you run it at a lower dimension, say D=5, then the BIC score should
correctly indentify the number of dimension.

So when using GMM, you must make sure that dimension<<samples.
"""

from numpy import *
from matplotlib.pylab import *
import pypr.clustering.gmm as gmm
import pypr.stattest as stattest

D = 5
K_orig = 5
samples_pr_cluster = 100

cen_lst = []
cov_lst = []

# Generate cluster centers, covariance, and mixing coefficients:
sigma_scl = 0.1
X = np.zeros((samples_pr_cluster*K_orig, D))
for k in range(K_orig):
    mu = np.random.randn(D)
    sigma = np.eye(D)*sigma_scl
    cen_lst.append(mu)
    cov_lst.append(sigma)
mc = np.ones(K_orig) / K_orig # All clusters equally probable

# Sample from the mixture:
N = 1000
X = gmm.sample_gaussian_mixture(cen_lst, cov_lst, mc, samples=N)

K_range = range(2, 10)
runs = 10
bic_table = np.zeros((len(K_range), runs))
for K_idx, K in enumerate(K_range):
    print "Clustering for K=%d" % K
    for i in range(runs):
        cluster_init_kw = {'cluster_init':'sample', 'max_init_iter':5, \
            'cov_init':'var', 'verbose':True}
        cen_lst, cov_lst, p_k, logL = gmm.em_gm(X, K = K, max_iter = 1000, \
            delta_stop=1e-2, init_kw=cluster_init_kw, verbose=True, max_tries=10)
        bic = stattest.bic_gmm(logL, N, D, K)
        bic_table[K_idx, i] = bic

plot(K_range, bic_table)
xlabel('K')
ylabel('BIC score')
title('True K=%d, dim=%d') % (K_orig, D)

