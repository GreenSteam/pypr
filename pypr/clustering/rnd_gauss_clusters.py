
import numpy as np


def rnd_gauss_clusters(centroids, ccov, samples):
    """
    Sample from a set of N Gaussian clusters (equally likely).

    centroids : List of cluster centers - [ [x1,y1,..],..,[xN, yN,..] ]
    ccov      : List of cluster co-variances DxD matrices

    Example:    centroids=[np.array([10,10])]
                ccov=[np.array([[1,0],[0,1]])]
                samples = 10
                rnd_gauss_clusters(centroids, ccov, samples)

    Returns a samples x D matrix.
    """
    cc = centroids
    
    # Determin number of clusters
    if (len(cc) != len(ccov)):
        raise ValueError('Lengths of centrodids and ccov must match')
    N = len(cc)
    
    D = len(cc[0]) # Determin dimensionality

    res = np.zeros((samples, D)); # For storing the result
    sc = np.random.randint(0, N, samples) # Randomly select clusters

    for cluster_no in range(N):
        s = np.random.multivariate_normal( \
                cc[cluster_no], ccov[cluster_no], samples)
        res[sc==cluster_no] = s[sc==cluster_no] 
        
    return res
