
import copy

# Based on code from Heina Hognason 2010-12-9:

def wrap_model(X0, functions, **indices): #, norm_x, norm_t, ann, **indices):
    """
    Wrap a chain of function calls. It is typical to call a series of functions
    to obtain a result from a model. For example a preprocessing function, 
    a model call, and a post-processing function. X0 is a defalt input for the
    function chain.

    Parameters
    ----------
    X0 : np array
        Default input for the chain. Samples row-wise, dimensions column-wise.
    functions : list of functions
        The list is called from left to right. For exaple [a,b,c] would result
        in c(b(a(X0)))
    **indices : dictionary, optional
        Specify a names for the colums in X0 or the input. For example saying
        that the first colums is called speed: {'speed':0}

    Returns
    -------
    eval_wrap : function
        A function for evaluation the wrapped network,

    Example
    -------
    ::

    example here

    """
    saved_X = X0.copy()

    def eval_wrap(*X, **values):
        """
        Evaluates a chain of functions.

        Parameters
        ----------
        *X : np array, optional
            Samples row-wise, inputs/dimensions column-wise. If specified
            this `X` will be used instead of the original X0 given in the
            wrapper constructor.
        **values : dictionary, optional
            Specify input name and the value it should be set to, for example
            one could say "Speed=1.0" to set all the speed column to 1.0.

        Examples
        --------
        eval_wrap(X)
        eval_wrap(waterspeed=20)
        eval_wrap(X, waterspeed=20)

        Returns
        -------
        Y : np array
            Normalized output from ANN.

        """
        if len(X)==0:
            array_x = saved_X.copy()
        elif len(X)==1:
            array_x = X[0].copy()
        else:
            raise Exception("Couldn't understand arguments.")
        for arg in values:
            index = indices[arg]
            value = values[arg]
            array_x[:,index] = value
        res = array_x
        for f in functions:
            res = f(res)
        return res
        #return norm_t.invtransform(ann.forward(norm_x.transform(array_x)))

    return eval_wrap

