import numpy as np

class Normalizer:
    """
    Rescale and shift each feature individually, so that it gets a mean
    of zero, and a unity standard deviation. The scaling and shifting
    are based upon `X`
    
    Parameters
    ----------
    X : np array
        An array with samples row wise and features/inputs column wise.
    handleStd0 : bool, optional
        True (default) or False. If true, the standard deviation
        is set to 1.0 if it is found to be 0, otherwise an exception is
        thrown in this case.
    """
    
    def __init__(self, X, handleStd0 = True):
        """
        Rescale and shift each feature individually, so that it gets a mean
        of zero, and a unity standard deviation. The scaling and shifting
        are based upon `X`
        
        Parameters
        ----------
        X : np array
            An array with samples row wise and features/inputs column wise.
        handleStd0 : bool, optional
            True (default) or False. If true, the standard deviation
            is set to 1.0 if it is found to be 0, otherwise an exception is
            thrown in this case.
        """
        r, c = np.shape(X)
        self.m = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        if (handleStd0):
            # If the standard deviation is zero then set it to one
            self.std[self.std==0] = 1 # added by SV 19.04.2010: replaced unnecessary for loop
        else: #otherwise throw exception
            if np.any(self.std==0): # added by SV 19.04.2010: replaced unnecessary for loop
                i = np.where(self.std==0)[0]
                raise ValueError("Standard deviation of input %d is zero" % i)
    
    def transform(self, X):
        """
        Find the normalized values for the samples in X, so that each input 
        will have zero mean and unit standard deviation.

        Parameters
        ----------
        X : np array
            An array with samples row wise and features/inputs column wise.
        Returns
        -------
        iX : np arrray
            Normalized version of `X`
        """
        r, c = np.shape(X)
        #iX = X.copy()
        if (c!=len(self.m)):
            raise Exception("Size mismatch.")
        #for i in range(0, c):
        #    iX[:,i] = (iX[:,i] - self.m[i]) / self.std[i]
        iX = (X - self.m) / self.std # added by SV 20.04.2010: replaced unnecessary for loop
        return iX
    
    def invtransform(self, iX):
        """
        Find the inverse transformation.

        Parameters
        ----------
        iX : np array
            An array with normalized samples row wise and features/inputs
            column wise.

        Returns
        -------
        X : np array
            The inverse transformation of the values given in `iX`.
        """
        r, c = np.shape(iX)
        #X = iX.copy()
        if (c!=len(self.m)):
            raise Exception("Size mismatch.")
        #for i in range(0, c):
        #    X[:,i] = X[:,i] * self.std[i] + self.m[i]
        X = iX * self.std + self.m # added by SV 20.04.2010: replaced unnecessary for loop
        return X


