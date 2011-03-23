
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

sum_of_squares = (_sum_of_squares, _sum_of_squares_d)


class WeightDecay():
    """
    Used to add a weight decay term to the networkd's error function.

    The penalty is calculated as the sum of the squared network weights
    multiplied by a given factor. This error is added to the error function.
    
    This is implemented as a class, as we need a reference to the network, or
    more precisely its weights.
    """    
    
    def __init__(self, ann, error_func):
        """
        Weight decay error function class.
        This class proviced a new error function, that includes the weight
        decay term. The new error function is obtained by calling the
        `get_ef()` method.

        Parameters
        ----------
        ann : ANN
            Use the weights from this network to estimate the penalty.
        error_func : tuple
            A tuple containing the orignal error function and its derivative.
            The weigth decay is added to this error function.
            
        Returns
        -------
        """
        self.ann = ann
        self.error_func = error_func
        self.v = 0.1
        
    def set_weight_decay_factor(v):
        self.v = v

    def eval_err(self, y, t):
        """
        Evaluate the error function, adding the weight decay term.
        Parameters
        ----------
        y: np array
        t: np array

        Returns
        -------
        err : scalar
            Error value
        """
        ef_res = self.error_func[0](y, t)
        sum_of_square_weights = np.sum(self.ann.get_flat_weights()**2)
        return ef_res + 0.5 * self.v * sum_of_square_weights
    
    def eval_err_d(self, y, t):
        return self.error_func[1](y, t)
    
    def get_ef(self):
        """
        Returns
        -------
        ef : function
            Returns the new error function, where that weight decay is added.
        """
        return (self.eval_err, self.eval_err_d)

    
