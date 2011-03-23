
import copy

# Based on code from Heina Hognason 2010-12-9:

def wrap_model(X0, functions, **indices): #, norm_x, norm_t, ann, **indices):
    """
    Wrap a Artificial Neural Network with a normalizer for the inputs, a
    normalizer for the output, and named indices for variable that can
    be set in the wrapper.

    Parameters
    ----------
    X0 : np array
        Default input for network. Samples row-wise, dimensions column-wise.
    norm_x : Normalizer
        Normalizer for network inputs.
    norm_t : Normalizer
        Normalizer for network outputs.        
    ann : ANN
        Artificial neural network to wrap.
    **indices : dictionary, optional
        Specify an index for each variable that can be set by the wrapper.

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
        Evaluate normalized output for an ANN with with a normalized X as input.

        Parameters
        ----------
        *X : np array, optional
            Samples row-wise, inputs/dimensions column-wise. If specified
            this `X` will be used instead of the original given in the
            wrapper constructor.
        **values : dictionary, optional
            Specify input name and the value it should be set to, for example
            one could say "Speed=1.0" to set the speed column to 1.0.

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

