
# Activation functions

import numpy as np

"""
The activation functions are represented as a tuple of python functions. The
first function in the tuple is the activation function itself, and the second
function in the tuple is the derivative of the activation function.
"""

def _lin(x):
    """
    """
    return x

def _lin_d(x):
    """
    """
    return 1

def _tanh(x):
    """
    """
    return np.tanh(x)

def _tanh_d(x):
    """
    """
    #return np.ones(np.shape(x)) - (np.tanh(x) * np.tanh(x))
    return 1.0 - x * x  # TODO: Investigate this!!!!

def _sigmoid(x):
    """
    """
    return 1 / (1 + np.exp(-x))

def _sigmoid_d(x):
    """
    """
    ga = _sigmoid(x)
    #return ga * (1-ga) hmm again
    return x * (1 - x)

lin = (_lin, _lin_d)
tanh = (_tanh, _tanh_d)
sigmoid = (_sigmoid, _sigmoid_d)
