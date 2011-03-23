#
# Data Driven Model
#

import pypr.ann.error_functions as ef # INCLUDE in ann output!! NOT DONE YET
import numpy as np

class ModelWithData(object):
    """
    The ModelWithData class encapsulates data and a model in one class, allowing easy use of new or existing
    algorithms, such as error function minimizers.
    TODO: Make more general, should be blind to x and t (just use 'data')
    """
    
    def __init__(self, model, trainX, trainT):
        """
        A model and a training set should at least be provided for creating a
        data driven model.
        """
        self.model = model
        self.trainX = trainX
        self.trainT = trainT

    def get_parameters(self):
        """
        Returns an row vector with all controllable weights in the model.
        """
        return self.model.get_flat_weights()

    def set_parameters(self, weights):
        """
        Sets the weights of the network.
        """
        self.model.set_flat_weights(weights)
    
    def err_func(self, weights):
        error_func = self.model.get_error_func()
        self.model.set_flat_weights(weights)
        y = self.model.forward(self.trainX)
        # Weight decay:        
        return error_func[0](y, self.trainT)
    
    def err_func_d(self, weights):
        #error_func = self.model.get_error_func()
        self.model.set_flat_weights(weights)
        dw = self.model.gradient(self.trainX, self.trainT)
        init = True
        for w in dw:
            if (init):
                res = w.flatten()
                init = False
            else:
                res = np.concatenate((res, w.flatten()), axis=1)
        #weight decay:        
        #return np.atleast_2d(res)
        return res
