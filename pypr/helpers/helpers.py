import numpy as np

def shuffle(A, axis = 0):
    """
    Returns a shuffled version of A. If axis = 0, then the rows are shuffled,
    if axis = 1 then the columns are shuffled.
    """
    if (axis!=0) and (axis!=1):
        raise ValueError, "Axis argument must be 0 or 1"
    r, c = np.shape(A)
    res = np.ones((r, c), dtype = A.dtype)
    if axis == 0:
        sr = range(0, r)
    else:
        sr = range(0, c)
    np.random.shuffle(sr)
    if axis == 0:
        res = A[sr, :]
    else:
        res = A[:, sr]
    return res

def shuffle_rows(A):
    """
    Returns an matrix with the rows in A shuffled. A must be 2d.
    """
    return shuffle(A, axis=0)

def MSE(y, t, axis=None):
    """Returns the Mean Square Error (MSE).
    """
    return np.mean((y-t)**2, axis=axis)

def RMSE(y, t, axis=None):
    """Returns the Root Mean Square Error (RMSE).
    """
    return np.sqrt(MSE(y, t, axis=axis))

def k_fold_cross_validation(X, K, k, inv=False):
    """K-fold cross-validation.

    Usage:
    X : NxD array
        Input data (N samples row-wise)
    K : int
        Numer of folds to make.
    k : int
        Fold number to leave out. Ranges from [0; K-1]
    inv : bool
        If true, then return the left out fold instead.
    """
    N = X.shape[0]
    d = np.linspace(0, N, K+1)
    d = np.array(d, dtype=np.int)
    if inv:
        return X[d[k]:d[k+1],:]
    else:
        X1 = X[d[0]:d[k], :]
        X2 = X[d[k+1]:, :]
        return np.concatenate((X1, X2), axis=0)


