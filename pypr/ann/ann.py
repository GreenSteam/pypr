import numpy as np
import numpy.random as rnd
import activation_functions as af
import error_functions as ef

#
# A simple feedforward multilayer neural network
#
# Please look at the style guide: http://www.python.org/dev/peps/pep-0008/
#
#
# TODO: Make sure no weights are zero


class ANN():
    """
    A simple implementation of a feed forward neural network.
    """
    
    def __init__(self, nodes, afunc=[], errorfunc=None):
        """
        Create an Artificial Neural Network (ANN)
        
        Parameters
        ----------
        nodes : list
            A list containing the number of nodes between each layer of
            weights (from input to output). E.g. [2,2,1] would result in
            a two input, one output network, with two hidden nodes.
        afunc : list, optional
            An activation function can be specified for each layer of 
            weights. The length of a `afunc` must be one less than the
            length of `nodes`. The activation functions are defined in
            the file activation_function.py.
            If not specified, then a tanh activation function will be used
            for all the layers, except the output layer, where a linear is
            used.
                    
        A basis note, which has a constant output of 1.0, is added for all
        node layers except the output layer.
        
        """
        self.weights = self._create_weights(nodes)
        if errorfunc==None:
            self._error_func = ef.sum_of_squares # Default
        else:
            self._error_func = errorfunc
        if len(afunc)==0:
            self.act_funcs = []
            for i in range(0, len(nodes)-2):
                self.act_funcs.append(af.tanh)
            self.act_funcs.append(af.lin)
        else:
            self.act_funcs = afunc
            
    def get_error_func(self):
        """
        Returns
        -------
        error_func : function
            The error function. This function can be overwritten, so for example
            a network with weight decay will return the error function with the
            weight decay penality.
        """
        return self._error_func

    def get_base_error_func(self):
        """
        Returns
        -------
        error_func : function
            This is the error function which back propagation uses. Should
            normally not be overwritten. This error function does not include
            the weight decay.
        """
        return self._error_func
        
    def _create_weights(self, nodes): #inputsNum, hiddenNum, outputsNum):
        # w(destination, source), + basis node
        resw = []
        for i in range(0, len(nodes)-1):
            N = nodes[i]
            w = rnd.randn(nodes[i+1], nodes[i]+1) / np.sqrt(N) # layer n+1, layer n            
            resw.append(w)
        return resw
    
    def get_num_layers(self):
        """
        Returns
        -------
        Nw : int
            Number of weights layers in the network.
        """
        return len(self.weights)

    def forward_get_all(self, inputs):
        """
        Returns the outputs from all layers, from input to output.
        
        Parameters
        ----------
        inputs : NxD np array
            An array with N samples with D dimensions (features). For example
            an input with 2 samples and 3 dimensions:

                inputs = array([[1,2,3], 
                                [3,4,5]])

        Returns
        -------
            result : list
                A list of 2d np arrays. The all the arrays will have N rows,
                the number of outputs for each array will depend on the
                of nodes in that particular layer.
        """
        w = self.weights # shorthand
        samples,d = np.shape(inputs)
        res = []
        x = np.concatenate((inputs, np.array([np.ones(samples)]).T), axis=1)
        for i in range(0, self.get_num_layers()):
            af = self.act_funcs[i][0]
            y = af(np.dot(x, w[i].T))
            res.append(y)
            if i+1 != self.get_num_layers():
                x = np.concatenate((y, np.array([np.ones(samples)]).T), axis=1)
        return res
    
    def forward(self, inputs):
        """
        Returns the output from the output layer.
            
        Parameters
        ----------
        inputs : NxD np array
            An array with N samples with D dimensions (features). For example
            an input with 2 samples and 3 dimensions:

                inputs = array([[1,2,3], 
                                [3,4,5]])

        Returns
        -------
            result : np array
                A NxD' array, where D' is the number of outputs of the
                network.
        """
        res = self.forward_get_all(inputs)
        return res[-1]
    
    def get_weight_copy(self):
        """
        Returns
        -------
            weights : list
                Copy of the current weights. The length of the list is equal
                to the number of hidden layers.
        """
        res = []
        for w in self.weights:
            res.append(w.copy())
        return res

    def gradient(self, inputs, targets,\
            errorfunc = None):
        """
        Calculate the derivatives of the error function. Return a matrix
        corresponding to each weight matrix.

        Parameters
        ----------
        inputs : NxD np array
            The inputs are given as, N, rows, with D features/dimensions.
        targets : NxT np array
            The N targets corresponding to the inputs. The number of outputs
            is given by T.
  
        Returns
        -------
        gradient : list
            List of gradient matrices.
        """
        if errorfunc is None:
            errorfunc = self.get_base_error_func()
        err_func = errorfunc[0]
        err_func_d = errorfunc[1] # Derivative of error function
        # targets: column of target
        #layer_outputs = forwardAllLayers(w, inputs)
        res = []
		#fwd_res is a list of arrays (length=number of layers in ann), each of which has shape [samples x output dimension]
        fwd_res = self.forward_get_all(inputs)
        samples, tmp = np.shape(inputs)
        #Handle the output layer as an special case:
        y = fwd_res[-1]
        af_d = self.act_funcs[-1][1]
        ef_d = errorfunc[1]
        delta = af_d(y) * ef_d(y, targets)  #(-(targets - y)) # TODO: costum error func, added af
        #delta = -(targets - y) # TODO: costum error func, and af
        dW = np.dot(delta.T, np.concatenate((fwd_res[-2] \
                    , np.ones([samples,1])), axis=1))
        res.insert(0, dW)
        for i in range(self.get_num_layers()-1,0,-1):
            af_d = self.act_funcs[i-1][1]
            r,c = np.shape(self.weights[i])
            #delta = (1.0-fwd_res[i-1]*fwd_res[i-1]) * (np.dot(delta,self.weights[i][:, 0:c-1]))
            delta =  af_d(fwd_res[i-1])* (np.dot(delta,self.weights[i][:, 0:c-1]))
            if i==1:
                dW = np.dot(delta.T, \
                    np.concatenate((inputs, \
                                    np.ones([samples,1.0])), axis=1))
            else:
                dW = np.dot(delta.T, \
                    np.concatenate((fwd_res[i-2], \
                                    np.ones([samples,1.0])), axis=1))
            res.insert(0, dW)
        return res
    
    def gradient_descent_train(self, inputs, targets, eta=0.001, maxitr=20000):
        """
        Train the network using the gradient descent method. This method
        shouldn't relly be used. Use the more advanced methods.

        Parameters
        ----------
        inputs : NxD np array
            The inputs are given as, N, rows, with D features/dimensions.
        targets : NxT np array
            The N targets corresponding to the inputs. The number of outputs
            is given by T.

        For example one could use the XOR function as an example:
            inputs = array([[0,0],         
                            [0,1],                          
                            [1,0],                         
                            [1,1]])                           
            targets = array([[0],[1],[1],[0]])    
        """
        for i in range(0, maxitr):
            dW = self.gradient(inputs, targets)
            for i in range(0, self.get_num_layers()):
                self.weights[i]= self.weights[i] - eta * dW[i]
        return
    
    def get_af_summed_weighs(self):
        """
        Returns
        -------
        sum : scalar
            Returns the sum of all the weights to the input of a note's
            activation function.
        """
        res = []
        for w in self.weights:
            wsr = np.c_[sum(w, axis=1)]
            res.append(wsr)
        return res
    
    def get_flat_weights(self):
        """
        Returns
        -------
        W : 1d np array
            Returns all the weights of the network in a vector form. From
            input to output layer. This is useful for the optimization
            methods, which normally only operate on an 1d array.
        """
        init = True
        for w in self.weights:
            if (init):
                res = w.flatten()
                init = False
            else:
                res = np.concatenate((res, w.flatten()), axis=1)
        #return np.atleast_2d(res)
        return res
    
    def set_flat_weights(self, W):
        """
        Set the weights of the network according to a vector of weights.

        Parameters
        ----------
        W : 1d np array
            W must have the correct length, otherwise it will not work.
        """
        idxStart = 0
        if W.ndim==2:
            rW, cW = W.shape
            if cW==1 or rW==1: W = W.flatten() # Could be a 2d column or row
            else:
                raise "Only 1-dim arrays please"
        if W.ndim>2:
                raise "Only 1-dim arrays please"
        for i in xrange(0, len(self.weights)):
            r, c = np.shape(self.weights[i])
            w = W[idxStart:r*c+idxStart]
            idxStart = idxStart + r * c
            w = w.reshape(r, c)
            self.weights[i] = w
        return

    def find_weight(self, flatWno):
        """
        Parameters
        ----------
        flatWno : int
            The number of the weight in a flattened vector

        Returns
        -------
        pos : tuple        
            Returns the layer number, row, and column where weight is
            found - (layer, r, c)

        This is not very fast, so do not use it for anything but small
        examples or testing.
            """
        curLay = 0
        no = flatWno
        while np.size(self.weights[curLay]) <= no:
            no -= np.size(self.weights[curLay])
            curLay += 1
        rw, cln = np.shape(self.weights[curLay])
        r = int(np.floor(no / cln))
        c = no % cln
        return (curLay, r, c)
    
    def find_flat_weight_no(self, layerNo, r, c):
        """
        Returns the corresponding flat weight index to layer and weight index.
        This is not very fast, so do not use it for anything but small examples
        or testing.
        """
        res = 0
        for i in range(0, layerNo):
            res += np.size(self.weights[i])
        rw, cln = np.shape(self.weights[layerNo])
        return res + cln * r + c

    def set_flat_weight(self, flatWno, value):
        """
        Sets the value of the weight with flat weight index.
        """
        (layer, r, c) = self.find_weight(flatWno)
        self.weights[layer][r,c] = value
        return

    def get_flat_weight(self, flatWno):
        """
        Returns the value of the weight corresponding to flat weight index.
        """
        (layer, r, c) = self.find_weight(flatWno)
        return self.weights[layer][r,c]

    def _gradient_finite_difference(self, inputs, targets, \
                                errorfunc = None):
        """
        Gives the same result as the gradient method. But this is an in-
        efficient implementation, and should not be used in practice. It is
        mostly included for testing reasons, and for curious people:)
        
        This method calculates the derivative of the error function with
        respect to each weight numerically, by using the Newton's difference
        quotient.
        
        By default the networks preferred error function is used.
        """
        if errorfunc is None:
            errorfunc = self.get_base_error_func()
        delta = 1e-6
        error_func = errorfunc[0]
        res = []
        for w in self.weights:
            r, c = np.shape(w)
            layerres = np.zeros(np.shape(w))
            origErr = error_func(self.forward(inputs), targets)
            for i in xrange(0, r):
                for j in xrange(0, c):
                    oldval = w[i,j]
                    w[i,j] = w[i,j]+delta;
                    newErr = error_func(self.forward(inputs), targets)
                    w[i,j] = oldval
                    layerres[i,j] = sum((newErr - origErr) / delta)
            res.append(layerres)
        return res
    
    def _hessian_finite_difference(self, inputs, targets, \
                                errorfunc = None):
        """
        Calculates the second order derivatives using finite difference. For each
        weight pair it needs to evaluate the feed forward network four times, 
        and hence requires O(W^3) operations, and hence is very slow. It is only
        used for testing purposes.
        """
        e = 0.0001
        if errorfunc is None:
            errorfunc = self.get_base_error_func()
        flatw = self.get_flat_weights()
        N = np.size(flatw)
        res = np.zeros((N, N))
        for i in range(0, N):
            for j in range(0, N):
                wi_orig = self.get_flat_weight(i)
                wj_orig = self.get_flat_weight(j)
                #
                self.set_flat_weight(i, wi_orig + e)
                self.set_flat_weight(j, wj_orig + e)
                E1 = error_func(self.forward(inputs), targets)
                #
                self.set_flat_weight(i, wi_orig + e)
                self.set_flat_weight(j, wj_orig - e)
                E2 = error_func(self.forward(inputs), targets)
                #
                self.set_flat_weight(i, wi_orig - e)
                self.set_flat_weight(j, wj_orig + e)
                E3 = error_func(self.forward(inputs), targets)
                #
                self.set_flat_weight(i, wi_orig - e)
                self.set_flat_weight(j, wj_orig - e)
                E4 = error_func(self.forward(inputs), targets)
                res[i, j] = (1/4*e^2) * (E1 - E2 - E3 - E4)
        return res
    
    
class WeightDecayANN(ANN):
    """This modifies the error function and the gradient to accommodate weight
decay, otherwise it works just like an ANN.
    """
    # TODO: Check gradient correctness
    def __init__(self, *args, **kwargs):
        if kwargs.has_key('weight_decay'):            
            self.v = kwargs[weightdecay]
            kwargs.pop('weight_decay')
        else:
            self.v = 0.1
        ANN.__init__(self, *args, **kwargs)
    
    def gradient(self, *args):
        grad = ANN.gradient(self, *args)
        oldgrad = grad
        for i in range(0, len(self.weights)):
            grad[i] = grad[i] + self.v * self.weights[i]
        #print oldgrad[0][0,0], " ", grad[0][0,0], " ", self.weights[i][0,0]
        return grad

    def error_with_weight_penalty(self, y, t):
        ef_res = self.get_base_error_func()[0](y, t)
        sum_of_square_weights = np.sum(self.get_flat_weights()**2)
        return ef_res + 0.5 * self.v * sum_of_square_weights
    
    def get_error_func(self):
        """Returns a modified error function with a weight decay penalty.
        """
        return (self.error_with_weight_penalty, None)
    
