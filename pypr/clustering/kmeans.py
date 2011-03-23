
#
# K-means
#
# Author: Joan Petur Petersen, December 2008
#

import numpy.random as rnd
import numpy as np


def find_distance(X, c):
    """
    Returns the euclidean distance for each sample to a cluster center
    
    Parameters
    ----------
    X : 2d np array
        Samples are row wise, variables column wise
    c : 1d np array or list
        Cluster center

    Returns
    -------
    dist : 2d np column array
        Distances is returned as a column vector.
    """
    dist = np.c_[ np.sum ( (X - c * np.ones(X.shape)) ** 2, axis=1) ]
    return np.sqrt(dist)

def find_intra_cluster_variance(X, m, cc):
    """
    Returns the intra-cluster variance.

    Parameters
    ----------
    X : 2d np array
        Samples are row wise, variables column wise
    m : list or 1d np array
        Cluster membership number, starts from zero
    cc : 2d np array
        Cluster centers row-wise, variables column-wise

    Returns
    -------
    dist : float
        Intra cluster variance
    """
    icv = 0.0
    h, w = cc.shape
    for i in range(h):
        icv += np.sum( find_distance(X[m==i, :], cc[i,:]) )
    return icv

def find_membership(X, cc):
    """
    Finds the closest cluster centroid for each sample, and returns cluster
    membership number.

    Parameters
    ----------
    X : 2d np array
        Samples are row wise, variables column wise
    cc : 2d np array
        Cluster centers row-wise, variables column-wise

    Returns
    -------
    m : 1d array
        cluster membership number, starts from zero
    """
    samples, w = np.shape(X)
    N, w2 = np.shape(cc)
    response = np.zeros((samples,1))
    smallest_dist = np.ones((samples,1)) * float('inf')
    for cno in range(N):
        d = find_distance(X, cc[cno,:])
        smaller_d = d < smallest_dist
        smallest_dist[smaller_d] = d[smaller_d]
        response[smaller_d] = cno
    return response.flatten()

def find_centroids(X, K, m):
    """
    Find centroids based on sample cluster assignments.

    Parameters
    ----------
    X : KxD np array
        K data samples with D dimensionallity
    K : int
        Number of clusters
    m : 1d np array
        cluster membership number, starts from zero

    Returns
    -------
    cc : 2d np array
        A set of K cluster centroids in an K x D array, where D is the number
        of dimensions.
        If a cluster isn't assigned any points/samples, then it centroid will 
        consist of NaN's.
"""
    samples, w = X.shape
    cc = np.zeros( [K, w] )
    
    for i in range(K):
        cc[i,:] = np.mean(X[m==i,:], axis=0)
    return cc


def kmeans(X, K, iter=20, verbose = False, \
                cluster_init = 'sample', \
                delta_stop = None):
    """
    Cluster the samples in X using K-means into K clusters.
    The algorithm stops when no samples change cluster assignment in an
    itertaion.
    NOTE: You might need to change the default maximum number of of iterations
    `iter`, depending on the number of samples and clusters used.

    Parameters
    ----------
    X : KxD np array
        K data samples with D dimensionallity.
    K : int
        Number of clusters.
    iter : int, optional
        Number of iterations to run the k-means algorithm for.
    cluster_init : string, optional
        Centroid initialization. The available options are: 'sample' and 
        'box'. 'sample' selects random samples as initial centroids, and 
        'box' selects random values within the space bounded by a box 
        containing all the samples.
    delta_stop : float, optional
        Use *delta_stop* to stop the algorithm early. If the change in all
        variables in all centroids is changed less than *delta_stop* then the 
        algorithm stops.
    verbose : bool, optional
        Make it talk back.

    Returns
    -------
    m : 1d np array
        cluster membership number, starts from zero.
    cc : 2d np array
        A set of K cluster centroids in an K x D array, where D is the number
        of dimensions.
        If a cluster isn't assigned any points/samples, then it centroid will 
        consist of NaN's.
"""

    samples, w = np.shape(X)

    # Initialize the centroids
    if cluster_init == 'box':
        cc = rnd.rand(K, w)
        cc = cc * (np.max(X, axis=0) - np.min(X, axis=0)) + np.min(X, axis=0)
    elif cluster_init == 'sample':
        rowidx = rnd.random_integers(0, high=samples-1, size=(K))
        cc = X[rowidx, :]
    else:
        raise "Unknown initialization of k-means centroids."

    if verbose:
        print "Initial cluster centers: "
        print cc
    oldcc = cc.copy()

    for ct in range(0, iter):
        
        # Update membership
        membership = find_membership(X, cc)

        # Update cluster centers
        cc = find_centroids(X, K, membership)
        
        # Break if centers have not changed
        if ( np.sum(cc==oldcc) == np.sum(np.ones(cc.shape)) ):
            if verbose:
                print "Stopped after %i iterations, no more change." % ct
            break
        
        if delta_stop is not None:
            change = np.abs( np.nan_to_num(cc - oldcc) )
            print change
            if not(np.any(change > delta_stop)):
                if verbose:
                    print "Stopped after %i iterations," + \
                        " all changes below delta_stop." % ct
                break
        oldcc=cc.copy()

    if verbose:
        print "Final cluster centers: "
        print cc
    return membership, cc
