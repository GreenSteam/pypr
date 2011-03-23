
Statistic and Statistical Tests
================================

Bayesian Information Criterion
------------------------------
When estimating model parameters using maximum likelihood estimation, it is possible to increase the likelihood by adding parameters, which may result in overfitting. The BIC resolves this problem by introducing a penalty term for the number of parameters in the model. This penalty is larger in the BIC than in the related AIC. [wikibic]_.

A rough approximation [bisp2006]_ [calinon2007]_ of the Bayesian Information Criterion (BIC) is given by

.. math::
    \ln p(\mathcal{D}) \backsimeq \ln p(\mathcal{D}|\Theta_{MAP}) - \frac{1}{2} M \ln N

where :math:`p(\mathcal{D}|\Theta_{MAP})` is the likelihood of the data given model, and :math:`N` is the number of samples, and :math:`M` is the number of free parameters :math:`\Theta` in the model (omitted in equation for simplicity).

The number of free parameters is given by the model used.

The number of parameters in a *Gaussian Mixture Model* (GMM) with :math:`K` clusters and a full covariance matrix, can be found by counting the free parameters in the means and covariances, which should give [calinon2007]_

.. math::
    M_{GMM} = (K-1) + K(D+ \frac{1}{2} D(D+1))

the parameter :math:`D` specifies the number of dimensions.

There is an example  :ref:`bicexample` that gives an example of using the BIC for controlling the complexity of a Gaussian Mixture Model (GMM). The example generates a plot, which should look something like this:

.. image:: figures/bic.*

.. [bisp2006] Christopher M. Bishop. Pattern Recognition and Machine Learning (Infor- mation Science and Statistics). Springer-Verlag New York, Inc., Secaucus, NJ, USA, 2006.


.. [calinon2007] Sylvain Calinon, Florent Guenter, and Aude Billard. On learning, repre- senting, and generalizing a task in a humanoid robot. Systems, Man and Cybernetics, Part B, IEEE Transactions on, 37(2):286-298, 2007. http://programming-by-demonstration.org/papers/Calinon-JSMC2007.pdf

.. [wikibic] Bayesian information criterion. (2011, February 21). In Wikipedia, The Free Encyclopedia. Retrieved 11:57, March 1, 2011, from http://en.wikipedia.org/w/index.php?title=Bayesian_information_criterion&oldid=415136150


Akaike Information Criterion
----------------------------
To be done.


Sample Autocorrelation
----------------------

Supposed we have a time series given by :math:`\{x_1,\ldots,x_n\}` with :math:`n` samples. 
The sample autocorrelation is calculated as follows:

.. math::
    \hat{\rho}_k = \frac{\sum_{t=1}^{n-k} (x_{t}-\bar{x})(x_{t+k}-\bar{x})}{\sum_{t=1}^n (x_t-\bar{x})^2}

where :math:`\bar{x}` is the mean for :math:`x_t`, and :math:`\hat{\rho}_k` is the sample autocorrelation

..
    If the function :math:`R` is well-defined, its value must lie in the range [-1, 1], with 1 indicating perfect correlation and -1 indicating perfect anti-correlation.
..
    .. [wikiac] Autocorrelation. (2011, February 22). In Wikipedia, The Free Encyclopedia. Retrieved 10:56, February 25, 2011, from http://en.wikipedia.org/w/index.php?title=Autocorrelation&oldid=415272769

.. automodule:: pypr.stattest.ljungbox
   :members: sac

Ljung-Box Test
--------------

The Ljung-Box test is a type of statistical test of whether any of a group of autocorrelations of a time series are different from zero. Instead of testing randomness at each distinct lag, it tests the "overall" randomness based on a number of lags [wikiljungbox]_. 


The test statistic is:

.. math::
    Q = n\left(n+2\right)\sum_{k=1}^h\frac{\hat{\rho}^2_k}{n-k}

where :math:`n` is the sample size, :math:`\hat{\rho}_k` is the sample autocorrelation at lag :math:`k`, and :math:`h` is the number of lags being tested [wikiljungbox]_. This function is implemented in:

.. automodule:: pypr.stattest.ljungbox
   :members: ljungbox


For significance level :math:`\alpha`, the critical region for rejection of the hypothesis of randomness is [wikiljungbox]_.

.. math::
    Q > \chi_{1-\alpha,h}^2 


where :math:`\chi_{1-\alpha,h}^2` is the :math:`\alpha`-quantile of the chi-square distribution with :math:`h` degrees of freedom. 
The chi-square distribution can be found in ``scipy.stats.chi2``.

Example
^^^^^^^
This example uses the *sample autocorrelation*, ``acf(...)``, which also is defined in ``pypr.stattest`` module.

.. literalinclude:: ../../examples/ljungbox_demo.py

The example generates a sample autocorrelation for the sun spot data set, and calculates the Ljung-Box test statistics.

.. image:: figures/ljungbox.*

The output should look something similar to this::

    lag   p-value          Q    c-value   rejectH0
    1       0.164      1.935      2.706      False
    2       0.378      1.948      4.605      False
    3       0.542      2.148      6.251      False
    4       0.600      2.752      7.779      False
    5       0.718      2.884      9.236      False
    6       0.823      2.884     10.645      False
    7       0.895      2.885     12.017      False
    8       0.941      2.897     13.362      False
    9       0.966      2.948     14.684      False
    10      0.941      4.132     15.987      False
    11      0.888      5.781     17.275      False
    12      0.922      5.887     18.549      False
    13      0.724      9.625     19.812      False
    14      0.744     10.242     21.064      False
    15      0.756     10.949     22.307      False
    16      0.746     11.969     23.542      False
    17      0.801     11.979     24.769      False
    18      0.847     12.008     25.989      False
    19      0.885     12.020     27.204      False


.. [wikiljungbox] Ljung–Box test. (2011, February 17). In Wikipedia, The Free Encyclopedia. Retrieved 12:46, February 23, 2011, from http://en.wikipedia.org/w/index.php?title=Ljung%E2%80%93Box_test&oldid=414387240
.. [adres1766] http://adorio-research.org/wordpress/?p=1766
.. [mathworks-lbqtest] http://www.mathworks.com/help/toolbox/econ/lbqtest.html


Box-Pierce Test
---------------
The Ljung–Box test that we have just looked at is a preferred version of the Box–Pierce test, because the Box–Pierce statistic has poor performance in small samples [wikilboxpierce]_.

The test statistic is [cromwell1994]_:

.. math::
    Q = n \sum_{k=1}^h\hat{\rho}^2_k

The implementation of the Box-Pierce is incorporated into the Ljung-Box code, and can be used by setting the method argument, ``lbqtest(..., method='bp')``, when calling the Ljung-Box test.

.. [wikilboxpierce] Box–Pierce test. (2010, November 8). In Wikipedia, The Free Encyclopedia. Retrieved 15:52, February 24, 2011, from http://en.wikipedia.org/w/index.php?title=Box%E2%80%93Pierce_test&oldid=395462997
.. [cromwell1994] Univariate tests for time series models, Jeff B. Cromwell, Walter C. Labys, Michel Terraza, 1994

Application Programming Interface
---------------------------------

.. automodule:: pypr.stattest.ljungbox
   :members:

