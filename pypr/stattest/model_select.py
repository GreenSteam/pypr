
import numpy as np

def bic(logL, M, N):
    """
    Bayesian Information Criteria

    Parameters
    ----------
    LogL : scalar
        Log likelihood value from model and data
    M : int
        Number of free parameters
    N : int
        Number of samples

    Returns
    -------
    score : scalar
        The BIC value (LogL - 0.5 * M * np.log(N))
    """
    return logL - 0.5 * M * np.log(N)


def bic_gmm(logL, N, D, K):
    """Bayesian Information Criterion for a Gaussian Mixture Model

    Parameters
    ----------
    N : int
        Number of samples.
    D : int
        Number of dimensions.
    K : int
        Number of clusters.

    Returns
    -------
    bic_score : scalar
        BIC score.
    """
    M = (K-1) + K*(D+0.5*D*(D+1))
    bic_score = bic(logL, M, N)
    return bic_score

