import numpy as np
import normalizer as nrm

class PCA(nrm.Normalizer):
    """
    Principal Component Analysis (PCA)

    Parameters
    ----------
    X : np array
        Expects an array X with samples row wise and features column
        wise. The eigen values and normalization used by the PCA are
        based upon X.
    normalize : bool, optional
        Optional input normalize specifies if the input data X should
        be normalized before the PCA transform.
    whitening : bool, optional
        If whitening is true, then the outputs are scaled so that they
        have zero mean and unity standard deviation. (NOT IMPLEMENTED)
    """
    
    def __init__(self, X, normalize = True, whitening = False, **kwargs):
        """
        Principal Component Analysis (PCA)

        Parameters
        ----------
        X : np array
            Expects an array X with samples row wise and features column
            wise. The eigen values and normalization used by the PCA are
            based upon X.
        normalize : bool, optional
            Optional input normalize specifies if the input data X should
            be normalized before the PCA transform.
        whitening : bool, optional
            If whitening is true, then the outputs are scaled so that they
            have zero mean and unity standard deviation. (NOT IMPLEMENTED)
        """
        self.whitening = False
        self.normalize = normalize
        if (normalize):
            nrm.Normalizer.__init__(self, X, **kwargs)
            X = nrm.Normalizer.transform(self, X)
        cov = np.cov(X.T)
        w,v = np.linalg.eig(cov)
        # Sort the eigenvector after their eigenvalues
        sortedIdx = np.argsort(np.abs(w))
        sortedIdx = sortedIdx[::-1] # Reverse array (most important first)
        w = w[sortedIdx]
        v = v[:, sortedIdx]
        self.w = w
        self.v = v

    def get_eig(self):
        """
        Eigen values.

        Returns
        -------
        w : np array
            Eigenvalues (length equal to number of inputs in `X`, D)
        v : np array
            Eigenvectors in a DxD sized matrix. Vectors a columns.

            The eigenvectors (columns) are sorted after eigenvalues in descending
            order.
        """
        return self.w, self.v

    
    def transform(self, X, dim = 0, skipnormalization = False):
        """
        Projects `X` into the new vector space found by the PCA.

        Parameters
        ----------
        X : np array
            Samples to be projected by PCA. Samples row-wise, inputs column-
            wise.
        dim : int, optional
            The number of most significant dimensions to return. If dim is
            zero, then all dimensions are returned.

        Returns
        -------
        Z : np array 
            Returns new projection of data. Samples row-wise, and `dim`
            columns.
        """
        if (self.normalize) and not(skipnormalization):
            X = nrm.Normalizer.transform(self, X)
        #dimen = c
        #if not(dim is None):
        #    dimen = dim
        r, c = np.shape(self.v)
        if dim==0: dim = r
        v = self.v[:, :dim]  #-dim:
        return np.dot(X, v)
    
    
    def invtransform(self, Z, skipnormalization = False):
        """
        Returns Z projected back in to the original vector space.

        Parameters
        ----------
        Z : np array 
            Projected data. Samples row-wise.

        Returns
        -------
        X : np array
            `Z` transformed to original vector space.
        """
        r, c = np.shape(Z)
        Z2 = np.zeros((r, len(self.v)))
        #Z2[:,-c:] = Z
        Z2[:,:c] = Z
        X = np.dot(Z2, self.v.T)
        if (self.normalize) and not(skipnormalization):
            return nrm.Normalizer.invtransform(self, X)
        else:
            return X
    
    
