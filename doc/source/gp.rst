

Gaussian Process
================

A Gaussian process is defined by its mean and covariance functions :math:`m(\mathbf{x})` and :math:`k(\mathbf{x},\mathbf{x}')` respectivily

.. math::
   \begin{array}{c}
     m(\mathbf{x}) = \mathbb{E} \left [ f(\mathbf{x}) \right ], \\
     k(\mathbf{x},\mathbf{x}') = \mathbb{E} \left [
       \left ( f(\mathbf{x})-m(\mathbf{x})\right )
       \left ( f(\mathbf{x}')-m(\mathbf{x}')\right )
     \right ]
   \end{array}


and Gaussian Process (GP) can be written as

.. math::
    f(\mathbf{x}) \sim \mathcal{GP} = \left (  m(\mathbf{x}), k(\mathbf{x},\mathbf{x}') \right )

The free parameters in the covariance functions are called hyperparameters.


Regression
----------

When doing regression we are interested in finding the model outputs for a given set of inputs, and the confidence of the predictions.
We have training dataset


.. math::
    \left [
    \begin{array}{c}
     \mathbf{f} \\ \mathbf{f_{*}} 
    \end{array}
    \right ]
    \sim
    \mathcal{N}
    \left (
    0,
    \left [
     \begin{array}{cc}
     K(X,X) & K(X,X_{*}) \\
     K(X_{*},X) & K(X_{*},X_{*}) \\
     \end{array}
    \right ]
    \right )
    \sim
    \mathcal{N}
    \left (
    0,
    \left [
     \begin{array}{cc}
     K & K_{*}^T \\
     K_{*} & K_{{*}{*}} \\
     \end{array}
    \right ]
    \right )

For simplicity we have set :math:`K=K(X,X)`, :math:`K_{*}=K(X_{*},X)`, and :math:`K_{{*}{*}}=K(X_{*},X_{*})`.

.. math::
   \bar{f}_{*} = K_{*}K^{-1}\mathbf{y}

.. math::
 var(f_{*}) = K_{*}K^{-1}K_{*}^T


Finding the hyper-parameters
----------------------------

Application Programming Interface
---------------------------------
.. automodule:: pypr.gp.GaussianProcess
   :members:



