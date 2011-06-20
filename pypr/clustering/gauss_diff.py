
"""Difference measures between pairs of Gaussian distributions.

The functions are translated from the gaussDiv.R R-project package
Original licence is GPL.
See: http://cran.r-project.org/web/packages/gaussDiff/
"""

# Original package header:

######################################################
### gaussDiv.R
### implement multiple divergence measures to compare normal
### pdfs, i.e. similarity and dissimilarity measures
### Henning Rust                        Paris 18/02/09

### all the implemented similarity and dissimilarity measures
### are described in the chapter
### Dissimilatiry Measures for Probability Distributions
### in the book "Analysis of Symbolic Data" by Hans-Hermann Bock 

import numpy as np

def _maha(x, A, factor=False):
    """
    """
    Ainv = np.linalg.inv(A)
    d = np.dot(x.T, np.dot(Ainv, x))
    if factor:
        return 0.5*d
    else:
        return d

def maha(mu1, sigma1, mu2):
    """
    """
    return _maha(mu1-mu2, sigma1, factor=True)

def hellinger(mu1, sigma1, mu2, sigma2, s=0.5):
    """Hellinger distance similarity measure.

    Returns
    -------
    d : float
        0<d<1,
        d=1, if P=Q,
        d=0, if P, Q have disjoint supports
    """
    sigma1inv = np.linalg.inv(sigma1)
    sigma2inv = np.linalg.inv(sigma2)
    sigma1inv_sigma2 = np.dot(sigma1inv, sigma2)
    sigma2inv_sigma1 = np.dot(sigma2inv, sigma1)
    N = sigma1.shape[0]
    I = np.diag(np.ones(N))
    d = np.linalg.det(s*I+(1-s)*sigma1inv_sigma2)**(-s/2) *\
        np.linalg.det((1-s)*I+s*sigma2inv_sigma1)**(-(1-s)/2) *\
        np.exp(0.5*(_maha(s*np.dot(sigma2inv, mu2) + (1-s) *\
        np.dot(sigma1inv, mu1), s*sigma2inv + (1-s)*sigma1inv) -\
        s * _maha(mu2,sigma2)-(1-s)*_maha(mu1,sigma1)))
    return d

def KL(mu1, sigma1, mu2, sigma2):
    """Kullback-Leibler divergence
    """
    N = mu1.shape[0]
    sig1inv = np.linalg.inv(sigma1)
    d = _maha(mu1-mu2, sigma1) + np.sum(np.diag(np.dot(sig1inv,sigma2)-\
                                        np.diag(np.ones(N)))) +\
            np.log(np.linalg.det(sigma1)/np.linalg.det(sigma2))
    return d * 0.5


def hellinger_weighted(mu1, sigma1, pi1, mu2, sigma2, pi2):
    """The weighted Hellinger distance used in the thesis Gaussian Mixture 
    Regression and Classification (2004) by  H. G. Sung.

    Formula 3.9 on page 34.
    Note: Not sure if it works correctly.
    """
    sigma1norm = np.linalg.norm(sigma1)
    sigma2norm = np.linalg.norm(sigma2)
    X0 = np.zeros(mu1.shape)
    i = 2 * (sigma1norm**(1.0/4)) * (sigma2norm**(1.0/4)) * np.sqrt(2*np.pi) *\
        gmm.mulnormpdf(X0, mu1-mu2, 2*sigma1 + 2*sigma2)
    #return np.sqrt(pi1*pi2) * (1-2*i)
    return 1-i[0]

