
# Activation functions

import numpy as np

"""
The activation functions are represented as a tuple of python functions. The
first function in the tuple is the activation function itself, and the second
function in the tuple is the derivative of the activation function.

x is passed to the activation functions as a NxD np array, with N samples and
D outputs.
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
    return x * (1 - x)

def _squash(x):
    """
    TECHNICAL RESEARCH REPORT, A Better Activation Function for Artificial
    Neural Networks, by D.L. Elliott, T.R. 93-8
    """
    return x / (1 + np.abs(x))

def _squash_d(x):
    """
    """
    return (1-np.abs(x))**2

def _softmax(x):
    """
    """
    r = np.zeros(x.shape)
    d = np.sum(np.exp(x), axis=1)
    for i in range(x.shape[1]):
        r[:,i] = np.exp(x[:,i]) / d
    return r

def _softmax_d(y):
    """
    """
    #return y * (1 - y)
    return np.ones(y.shape)
    #return y

lin = (_lin, _lin_d)
tanh = (_tanh, _tanh_d)
sigmoid = (_sigmoid, _sigmoid_d)
squash = (_squash, _squash_d)
softmax = (_softmax, _softmax_d)

