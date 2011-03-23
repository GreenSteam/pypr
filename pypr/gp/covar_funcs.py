
import numpy as np
import scipy.spatial.distance as dist


# Covariance functions ----------------------------------------------------------
"""
The docstrings for all the derived covariance functions will be very similar
to the original ones in cfCovarianceFunction, and have therefore to a large
extent been omitted.
"""

class cfCovarianceFunction(object):
    """
    The base covariance function class.
    When creating a new covariance function, you should inherit this class.
    """
    def __init__(self):
        raise Exception, "You should not use this function, just extend it."
    def eval(self, x, xs=None):
        """
        Calculate the covariance matrix.

        Parameters
        ----------
        x : NxD np array
            The training set `x` is a NxD matrix, with N samples of
            dimensionallity D.
        xs : MxD np array, optional
            `xs` is the testset input, containing M samples. The
             dimensionallity of the test set must be D.

        Returns
        -------
        cov : NxN np array or a tuple
            Depending if `xs` is specified the output will differ.
              
        If `xs` is None, then the covariance matrix is returned. The matrix
        `x` is an (N x D) matrix, where there are N samples, and D 
        dimensions. The returned matrix (array) will be a (N x N) matrix.

        `xs` is used to specify a test set. If xs (M x D) is set, a tuple
        (Kssdiag, Ks) is returned. Kssdiag is a column vector with m self
        covariances of the test set. Ks is a (N x M) matrix containing the
        cross variance between the training and test data.
        """
        raise Exception, "You must implement the eval(self, x1, x2) function."

    def derivative(self, x, param_no):
        """
        Evaluate a partial derivative of the covariance function.

        Parameters
        ----------
        x : NxD np array
            Data to evaluate the covariance function on.
        param_no : int
            Parameter number to evaluate the partial derivative for.

        Returns
        -------
        der : NxN np array
            Returns the partial dirivatives with respect to the log of hyper-
            parameter number param_no. If `x` is a (N x D) matrix, then the
            returned matrix will be a (N x N) matrix.
        """
        raise Exception, "You must implement derivative(self, x, param_no)."

    def __add__(self, other):
        """
        You should not override this methods, it handles the addition of
        covariance functions.
        """
        return cfSum(self, other)
    
    def get_params(self):
        """
        Get the hyper-parameters.

        Returns
        -------
        hp : list
            Returns a list of hyperparameters. The order of the list is
            important.
        """
        raise Exception, "Please override this method."
    
    def set_params(self, log_parameters):
        """Set hyperparameters to the values given.

        Parameters:
        log_parameters : list
            Set the hyperparameters. The order of the list is important. The
            covariance functions' hyperparameters are set from left to right.
        """
        raise Exception, "Please override this method."
    
    def clear_temp(self):
        """
        This method clears all temporary variables used by the covariance functions.
        Should be called after finishing using the function, for example training.
        """
        pass

class cfSum(cfCovarianceFunction):
    """
    The purpose of this class is to represent an addition of two covariance
    functions.
    """
    def __init__(self, a, b):
        """
        Sum of two covariance functions.

        Parameters
        ----------
        a : cfCovarianceFunction
            Left hand side cf
        b : cfCovarianceFunction
            Right hand side cf

        Returns
        -------
        sum_obj : cfSum
            Object that handles sum of covariance functions.
        """
        self.a = a
        self.b = b
    
    def eval(self, x, xs=None):
        """See help(cfCovarianceFunction.eval)
        """
        a = self.a.eval(x, xs=xs)
        b = self.b.eval(x, xs=xs)
        if xs==None:
            return a + b
        else:        
            return (a[0]+b[0], a[1]+b[1])
        
    def derivative(self, x, param_no):
        """See help(cfCovarianceFunction.derivative)
        """
        par_cnt_a = len(self.a.get_params())
        if param_no<par_cnt_a:
            return self.a.derivative(x, param_no)
        else:
            return self.b.derivative(x, param_no-par_cnt_a)

    def get_params(self):
        """Returns a list of the log of the hyperparameters.
        """
        return self.a.get_params()+self.b.get_params()
    
    def set_params(self, log_params):
        """
        Set the log of the hyperparameters to log_parameters.
        The parameters in the left hand side covariance function are set to
        first part of the list, the right hand side covariance function to
        the rest.
        """
        no_a = len(self.a.get_params())
        self.a.set_params(log_params[:no_a])
        self.b.set_params(log_params[no_a:])

    def clear_temp(self):
        """
        This method clears all temporary variables used by both the
        covariance functions in the sum.
        """
        self.a.clear_temp()
        self.b.clear_temp()


class cfSquaredExponentialIso(cfCovarianceFunction):
    """
    The isotropic Squared Exponential covariance function.
    """
    def __init__(self, log_ell=0.0, log_sqrt_sf2=0.0):
        """
        The isotropic Squared Exponential covariance function.

        Parameters
        ----------
        log_ell : float, optional
            Log lenght scale scalar
        log_sqrt_sf2 : float, optional
            Log sqrt of signal variance    

        Returns
        -------
        cfSEIso : cfSquaredExponentialIso
        """
        self.logparam = [log_ell, log_sqrt_sf2]

    def eval(self, x, xs=None):
        """See help(cfCovarianceFunction.eval)
        """
        ell = np.exp(self.logparam[0]);
        sf2 = np.exp(2*self.logparam[1]);

        if xs==None:
            return sf2*np.exp(-sq_dist(x/ell)/2)
        else:
            Kssdiag = np.c_[np.diag(sf2*np.exp(-sq_dist(xs/ell)/2))]
            Ks = sf2*np.exp(-sq_dist(x/ell, xs/ell)/2)
            return (Kssdiag, Ks)
        
    def derivative(self, x, param_no):
        """See help(cfCovarianceFunction.derivative)
        """
        ell = np.exp(self.logparam[0]);
        sf2 = np.exp(2*self.logparam[1]);
        if param_no==0:
            return sf2*np.exp(-sq_dist(x/ell)/2)*sq_dist(x/ell)
        else:
            return 2*sf2*np.exp(-sq_dist(x/ell)/2)

    def get_params(self):
        """
        Returns a list of the log of the hyperparameters.

        Returns
        -------
        log_hyperparameters : list
            A list containing two elements
            log_ell        Log lenght scale scalar
            log_sqrt_sf2   Log sqrt of signal variance    
        """
        return self.logparam
    
    def set_params(self, log_parameters):
        """
        Set the log of the hyperparameters to log_parameters.

        Parameters
        ----------
        log_parameters : list
            Should be a list containing two float entries
            log_ell        Log lenght scale scalar
            log_sqrt_sf2   Log sqrt of signal variance    
        """
        self.logparam = log_parameters


class cfNoise(cfCovarianceFunction):
    """
    White noise.
    """
    def __init__(self, log_theta=-2.3026):
        """
        White noise.

        Parameters
        ----------
        log_theta : float, optional
            `log_theta` specifies the log noise variance. The variance
            s2 = exp(2*log_theta).
        """
        self.logparam = [log_theta]
        
    def eval(self, x, xs=None):
        """See help(cfCovarianceFunction.eval)
        """
        s2 = np.exp(2*self.logparam[0])
        if xs==None:
            return s2 * np.eye(len(x))
        else:
            return (np.c_[s2*np.ones(len(xs))], np.zeros((len(x),len(xs))))
            
    def derivative(self, x, param_no):
        """See help(cfCovarianceFunction.derivative)
        """
        s2 = np.exp(2*self.logparam[0])
        return 2*s2*np.eye(len(x))

    def get_params(self):
        """
        Get white noise hyperparameter

        Returns
        -------
        log_hyperparameter : list
            Returns a list containing one hyperparameter, which is the log of
            white noise variance.
        """
        return self.logparam
    
    def set_params(self, log_parameters):
        """
        Set the white noise hyperparameter.

        Parameters
        ----------
        log_parameters : list
            Set the gaussian noise hyperparameter. Must be a list with one
            element. The element is the log of the noise variance.
        """
        self.logparam = log_parameters

class cfJitter(cfCovarianceFunction):
    """
    Constant jitter, similar to noise, but cannot be changed.
    This can make the regression calculation better conditioned.
    """
    def __init__(self, log_theta=-9.21):
        """
        Contant jitter covariance function.

        Parameters
        ----------
        log_theta : float, optional
            Log_theta given in the natural logarithm.
            Exp(log_theta=-9.21) corresponds approximately to 10**-4
        """
        self.logparam = [log_theta]
        self.krnds = []        
        
    def eval(self, x, xs=None):
        """See help(cfCovarianceFunction.eval)
        """
        #if len(self.krnds)!=x.shape[0]:
        #    print "Created new Jitter.krnds!"
        #    self.krnds = np.random.randn(x.shape[0])*10**(self.logparam[0])

        if xs==None:
            #return np.eye(len(x)) * self.krnds
            return np.eye(len(x)) * np.exp(self.logparam[0])
        else:
            # Maybe the noise should be added here too:
            return (np.c_[np.zeros(len(xs))], np.zeros((len(x),len(xs))))

    def derivative(self, x, param_no):
        """See help(cfCovarianceFunction.derivative)
        """
        return 0.0*np.eye(len(x))

    def get_params(self):
        """Returns a list containing one element, which is the gaussian noise
        """
        return self.logparam
    
    def set_params(self, log_parameters):
        """Set the gaussian noise hyperparameter. Must be a list with one element.
        """
        self.logparam = log_parameters

class cfSquaredExponentialARD(cfCovarianceFunction):
    """
    The Squared Exponential covariance function with Automatic Relevance 
    Determination (ARD).
    """
    def __init__(self, log_ell_D=[0.0], log_sqrt_sf2=0.0):
        """
        The Squared Exponential covariance function with Automatic Relevance 
        Determination (ARD).

        Parameters
        ----------
        log_ell_D : list, optional
            A list with D (dimensions) log lenght scale scalars.
        log_sqrt_sf2 : list, optional
            Log sqrt of signal variance
        """
        self.logparam = log_ell_D + [log_sqrt_sf2]
        self.K = None

    def eval(self, x, xs=None):
        """See help(cfCovarianceFunction.eval)
        """
        D = len(self.get_params())-1
        ell = np.exp(self.logparam[0:D]);
        sf2 = np.exp(2*self.logparam[D]);
        if xs==None:
            t = np.dot(np.diag(1./ell),x.T)
            K = sf2*np.exp(-sq_dist(t.T)/2)
            return K
            #t = np.dot(np.diag(1./ell),x.T)
            #return sf2*exp(-sq_dist(t.T)/2);
        else:
            #Kssdiag = np.c_[np.diag(sf2*np.exp(-sq_dist(xs/ell)/2))]
            #Ks = sf2*np.exp(-sq_dist(x/ell, xs/ell)/2)
            Kssdiag = sf2*np.ones((len(xs),1))
            Ks = sf2*np.exp(-sq_dist( np.dot(np.diag(1./ell),x.T).T , np.dot(np.diag(1./ell),xs.T).T )/2)
            return (Kssdiag, Ks)
        
    def derivative(self, x, param_no):
        """See help(cfCovarianceFunction.derivative)
        """
        D = len(self.get_params())-1
        ell = np.exp(self.logparam[0:D]);
        sf2 = np.exp(2*self.logparam[D]);
        if self.K==None: # This is to save calculations
            t = np.dot(np.diag(1./ell),x.T)
            K = sf2*np.exp(-sq_dist(t.T)/2)
            self.K = K
            self.meanx = np.mean(x)
        else:
            if self.meanx==np.mean(x):
                K = self.K # Reuse from last calculation
            else:
                t = np.dot(np.diag(1./ell),x.T)
                K = sf2*np.exp(-sq_dist(t.T)/2)
                self.K = K
                self.meanx = np.mean(x)
        if param_no<D:
            return K*sq_dist(x[:,param_no].T/ell[param_no])
        else:
            return 2*K

    def get_params(self):
        """
        Get hyperparameters.

        Returns
        -------
        hyperparameters : list
            Returns a list of the log of the hyperparameters. The first 
            elements in the list are the length scale hyperparameters (ARD
            parameters) and the last is the log square signal variance.
        """
        return self.logparam
    
    def set_params(self, log_parameters):
        """Set the log of the hyperparameters

        Parameters
        ----------
        log_parameters : list
            A list with D (number of dimensions) log lenght scale scalars,
            and a log_sqrt_sf2 at the end, log sqrt of signal variance.
        """
        self.logparam = log_parameters
        self.K = None
        
    def clear_temp(self):
        """This method clears all temporary variables used by the covariance function.
        """
        self.K = None



def sq_dist(A, B=None):
    """
    """
    if B==None:
        return dist.squareform(dist.pdist(np.c_[A],'sqeuclidean'))
    else:
        # Find a better way to do this:
        C = np.concatenate((A,B), axis=0)
        D = dist.squareform(dist.pdist(np.c_[C],'sqeuclidean'))
        return D[:len(A),len(A):]

