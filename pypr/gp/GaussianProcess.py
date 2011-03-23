
import sys
import numpy as np
import scipy.optimize
import scipy.linalg as la
from pypr.optimization import minimize
from covar_funcs import *

class GaussianProcess:
    """
    Gaussian Process class.
    """

    def __init__(self, cf):
        """
        Initialize a Guassian Process object.

        Parameters
        ----------
        cf : cfCovarianceFunction or a class that implements it
            Covariance function, see cfCovarianceFunction for more help.

        Returns
        --------
        object: GaussianProcess
            An Gaussian Process object
        """
        # Do some input sanity checking:
        self.cf = cf
        self.krnds = []
        
    def fit_data(self, X, y):
        """
        Fit the hyper-parameters to the data.
        
        X   Input samples (2d array, samples row-wise)
        y   Function output (1d array)
        """
        # Do some input sanity checking:
        if X.shape[0]!=y.shape[0]:
            raise ValueError, "X and y must have the same number of samples."

        if X.ndim != 2:
            raise ValueError, "X must be a 2 dimensional array"

        if y.ndim != 1:
            raise ValueError, "y must be a 1 dimensional array"
 
        # If all ok
        self.X = X
        self.y = y

#    def find_covariance(self, xx=None):
#        """
#        Returns the co-variance matrix.
#        """
#        K = np.zeros((len(x), len(x)))
#        for i in range(len(x)):
#            for j in range(len(x)):
#                K[i, j] = cf(x[i], x[j])
#        if xx==None:
#            return K
#        Kx = np.zeros((len(x), len(xx)))
#        Kxx = np.zeros((len(xx), len(xx)))
#        for i in range(len(x)):
#            for j in range(len(xx)):
#                Kx[i, j] = cf(x[i], xx[j])
#        for i in range(len(xx)):
#            for j in range(len(xx)):
#                Kxx[i,j] = cf(xx[i], xx[j])
#        return K, Kx, Kxx
    
    def generate(self, x, rn=None):
        """
        Generate samples from the GP.

        Parameters
        ----------
        x : np array
            A 2d matrix with samples row-wise        
        rn : np array
            Provide your own random numbers (mostly just for testing)

        Return
        ------
        sampes : np array
            Samples from GP
        """
        if rn==None:
            rn = np.random.randn(len(x),1)
        return np.dot(np.linalg.cholesky(self.cf.eval(x)), rn)
    
    def find_likelihood_der(self, X, y):
        """
        Find the negative log likelihood and its partial derivatives.

        Parameters
        ----------

        Returns
        -------
        """
        n = len(X)
        K = self.cf.eval(X)

        #if len(self.krnds)!=K.shape[0]:
        #    print "Created new self.krnds!"
        #    self.krnds = np.random.randn(K.shape[0])*10**-6
        #K = K + np.eye(K.shape[0])*self.krnds        

        L = np.linalg.cholesky(K) # Problems using this on the cluster - bad scaling! Running time becomes really bad with large N. Solution: Update ATLAS
        #L = la.cholesky(K)
        #print np.linalg.solve(L.T, np.linalg.solve(L, y))
        #a = np.linalg.solve(L.T, np.linalg.solve(L, y))
        a = la.cho_solve((L, True), y)

        nll = 0.5*np.dot(y.T, a) + np.sum(np.log(np.diag(L))) + 0.5*n*np.log(2*np.pi)
        ders = np.zeros(len(self.cf.get_params()))
        #W = np.linalg.solve(L.T, np.linalg.solve(L, np.eye(n))) - a*a.T

        W = la.cho_solve((L, True), np.eye(n))  - a*a.T
        
        for i in range(len(self.cf.get_params())):
            ders[i] = np.sum(W*self.cf.derivative(X, i))/2
        return nll[0,0], ders
    
    
    def regression(self, X, y, XX, max_samples=1000):
        """
        Predict the y values for `XX` given the `X`, `y`, and the objects
        covariance function.

        Parameters
        ----------
        X : MxD np array
            training array containing M samples of dimension D
        y : 1-dimensional np array of length M
            training outputs
        XX : NxD np array
            An array containing N samples, with D inputs
        max_samples : int, optional
            Due to memory considerations, the inputs are evaluated part
            at a time. The maximum number of samples evaluated in each
            interation is given by `max_samples`

        Returns
        -------
        ys : np array of length N
            output of GPR
        s2 : np array of length N
            corresponding variance    
        """
        # We must avoid the covariance matrix getting to large. We cannot do so much
        # about X, but we can calculate a part of XX at a time.
        max_samples = 1000 # Maximum samples from XX to take
        blocks = XX.shape[0] / max_samples
        l = np.linspace(0, XX.shape[0], blocks+1)
        l = l.astype(int)
        if len(l)<2:
            l = [0, XX.shape[0]] # if there are few samples in XX then do everything
        ys_all = None
        s2_all = None
        for i in range(len(l)-1):
            XX2 = XX[l[i]:l[i+1],:]
            n = len(X)
            K = self.cf.eval(X)
            L = np.linalg.cholesky(K)
            #a = np.linalg.solve(L.T, np.linalg.solve(L, y))
            a = la.cho_solve((L, True), y)
            (Kssdiag, Ks) = self.cf.eval(X, XX2)
            ys = np.dot(Ks.T, a)
            try:
                v = la.solve_triangular(L, Ks, lower=True)
            except AttributeError:
                v = np.linalg.solve(L, Ks) # L is a lower-triangular matrix, so can do this faster
            s2 = Kssdiag - np.c_[np.sum(v*v, axis=0)]
            if ys_all==None:
                ys_all = ys
            else:
                ys_all = np.concatenate((ys_all, ys), axis=0)
            if s2_all==None:
                s2_all = s2
            else:
                s2_all = np.concatenate((s2_all, s2), axis=0)
        return ys_all, s2_all 


class GPR(object):
    """
    Gaussian Process for regression.
    """

    def __init__(self, GP, X, y, max_itr=200, callback=None,
                print_progress=False, no_train=False, mean_tol=None):
        """
        Create an GP regression object. The hyper-parameters will be found
        automatically (trained). The training will be based upon the
        dataset given by `X` and `y`.

        The difference between the GP and GPR, is that the GPR remembers
        the training data `X` and `y`.
 
        Parameters
        ----------
        GP : GaussianProcess object
            Gaussian Process with given covariance functions
        X : np array
            Input samples (2d array, samples row-wise)
        y : np array
            Function output (1d array)
        mean_tol: float, optional
            Specifies the tolerance that the mean check uses. Can be
            turned of setting to inf.
        print_progress : bool, optional
            If `print_progress` is true, the progress will be printed.
        
        A callback function can be specified, which is called for each
        iteration. The callback function has to be of the form: 
        def callback(gpr, hp) or def callback(self, gpr, hp) if an object
        method.

        Returns
        -------
        gpr : GPR object
            A GPR object
        """
        if mean_tol==None:
            self.mean_tol = 0.01
        else:
            self.mean_tol = mean_tol
        self.GP = GP
        self.X = X
        self.y = y
        self._itr = 0
        self.callback = callback
        self.print_progress = print_progress
        self.old_lh_res = None
        self.remember_lh = True
        self.max_itr = max_itr
        if no_train==False:
            self.find_hyperparameters()

    def find_hyperparameters(self, max_itr=None):
        """
        Find hyperparameters for the GP.
        """
        if max_itr!=None:
            self.max_itr = max_itr
        if self.print_progress==False:
            self._train(max_itr=self.max_itr, callback=self.callback)
        else:
            # print_progress will then call the user specified callback
            self._train(max_itr=self.max_itr, callback=self._print_progress)
        self.GP.cf.clear_temp()

            
    def f(self, params=None):
        """
        The function to minimize, which is the likihood in this case. This
        method is passed to the optimizer.

        Parameters
        ----------
        params : 1D np array, optional
            hyper-parameters to evaluate at. It not specified, then the
            current values in the covariance function are used.

        Returns
        -------
        nllikelihood : float
            Negative log likihood
        """
        # TODO:
        # Try to avoid some calculations (assumes X and y havn't changed):
        #if (self.remember_lh==True) and (self.old_lh_res!=None):
        #    if np.alltrue(self.old_lh_res['params']==params):
        #        return np.atleast_2d(self.old_lh_res['lh'][0])

        if params!=None:
            self.GP.cf.set_params(params.flatten().tolist())
        lh = (self.GP.find_likelihood_der(self.X, self.y))
        res = np.atleast_2d(lh[0])

        #Store results
        if self.remember_lh:
                self.old_lh_res = {'params':params, 'lh':lh}
        res = res.flatten()[0]
        #print "res=", res
        #print "res.__class__=", res.__class__
        #print "len(res)=",len(res)
        return res
    
    def df(self, params=None):
        """
        The partial derivatives of the function to minimize, which is the
        likihood in this case. This method is passed to the optimizer.

        Parameters
        ----------
        params : 1D np array, optional
            hyper-parameters to evaluate at. It not specified, then the
            current values in the covariance function are used.

        Returns
        -------
        der : np array
            array of partial derivatives
        """
        # TODO:
        # Try to avoid some calculations (assumes X and y havn't changed):
        #if (self.remember_lh==True) and (self.old_lh_res!=None):
        #    if np.alltrue(self.old_lh_res['params']==params):
        #        return np.atleast_2d(self.old_lh_res['lh'][1])

        if params!=None:
            self.GP.cf.set_params(params.flatten().tolist())
        lh = (self.GP.find_likelihood_der(self.X, self.y))
        der = lh[1]

        #Store results
        if self.remember_lh:
            self.old_lh_res = {'params':params, 'lh':lh}
        der = der.flatten()
        #print "der=", der
        #print "der.shape=", der
        #print "len(params)=",len(params)
        #print "der.__class__=", der.__class__
        return der
    
    def _train(self, max_itr, callback=None):
        """
        """
        # Check for zero mean and unit variance
        means = np.mean(self.X, axis=0)
        for m in means:
            if np.abs(m) > self.mean_tol:
                raise Exception( \
                    "The samples in X should have a zero mean. That is every column in X" +
                    " should have a zero mean. You can disable this check in the class" +
                    " constructor using mean_tol=inf. Means where found to be: " +
                    str(np.mean(self.X, axis=0)) )
        #np.seterr(invalid='raise')
        self.remember_lh = True
        self._itr = 0
        X0 = np.atleast_2d((self.GP.cf.get_params()))
        X0 = X0.flatten()
        minimize(X0, self.f, self.df, max_itr, callback=callback)
        #print "X0=", X0
        #scipy.optimize.fmin_cg(self.f, X0, disp=True, maxiter=max_itr, fprime=self.df, callback=callback)
        #scipy.optimize.fmin_bfgs(self.f, X0, fprime=self.df)
        #scipy.optimize.fmin_ncg(self.f, X0, fprime=self.df)
        self.GP.cf.clear_temp()
        self.remember_lh = False
        self.old_lh_res = None

    def _print_progress(self, params):
        print "Iteration number: ", self._itr
        print np.atleast_2d((self.GP.find_likelihood_der(self.X, self.y))[0])        
        self._itr += 1
        print "Parameters: ", params
        sys.stdout.flush() # python doesn't really like to write to stdout :)
        if self.callback!=None:
            self.callback(self, params)

    def predict(self, XX):
        """
        Predict the the output of the GPR for the inputs given in `XX`

        Parameters
        ----------
        XX : NxD np array
            An array containing N samples, with D inputs

        Returns
        -------
        res : np array
            An array of length N containing the outputs of the network            
        """
        return self.GP.regression(self.X, self.y, XX)
        

