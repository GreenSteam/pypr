
# Error functions

"""
An error function is represented as a tuple with a function for evaluating
networks error and its derivatives.
"""

import numpy as np

def _sum_of_squares(y, t):
    """
    """
    return np.array([0.5*np.sum((y-t)*(y-t))])

def _sum_of_squares_d(y, t):
    """
    """
    return (y - t)

def _entropic(y, t):
    """
    """
    res = 0    
    for k in range(y.shape[1]):
        res += np.sum(t[:,k]*np.log(y[:,k]))
    return np.array([-res])

def _entropic_d(y, t):
    """
    """
    return (y - t)

sum_of_squares = (_sum_of_squares, _sum_of_squares_d)
entropic = (_entropic, _entropic_d)

