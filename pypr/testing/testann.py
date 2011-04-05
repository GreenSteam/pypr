
import pypr.ann as ann
import pypr.ann.activation_functions as af
import pypr.ann.error_functions as ef
import random
import unittest
import numpy as np

class TestANN(unittest.TestCase):

    def setUp(self):
        # Not much here yet
        self.seq = range(10)


    def test_default_constructor(self):
        """
        Checks if the sizes of the weights are correct, and that the default
        activation function are correctly set.
        """
        a = ann.ANN([1,2,1])
        self.assertEqual(a.weights[0].shape, (2,2))
        self.assertEqual(a.weights[1].shape, (1,3))
        self.assertEqual(a.act_funcs, [af.tanh, af.lin])

    def test_forward_get_all(self):
        """
        Test the forwarding of an given input, given weights and outputs.
        """
        a = ann.ANN([2,2,1])
        a.weights = [np.array([[1., 2, 3], [8, 3, 5]]) / 10, \
                     np.array([[ 5. ,  2 ,  5]]) / 10]
        inputs = np.array([[1,2]])
        r = a.forward_get_all(inputs)
        err = np.sum(\
            np.power(r[0] - np.array([[ 0.66403677,  0.95623746]]), 2) )
        err += np.sum(\
            np.power(r[1] - np.array([[ 1.02326588]]), 2) )
        self.assertAlmostEqual(err, 0.0)
        
    def test_gradient(self):
        """
        Test if the gradient is calculated correctly.
        This is done using the finite difference gradient.
        Multiple, two actually, outputs are tested :)
        """
        a = ann.ANN([2,5,7,3,2], [af.sigmoid, af.tanh, af.lin, af.squash])
        inputs = np.array([[1, 4], [2, 3], [3, 1]])
        targets = np.array([[0.4, 0.6], [0.5, 0.5], [0.8, 0.2]])
        dw1 = a.gradient(inputs, targets)
        dw2 = a._gradient_finite_difference(inputs, targets)
        totsum = calc_weight_abs_diff(dw1, dw2)
        self.assertAlmostEqual(totsum, 0.0, places=3)

    def test_softmax_gradient(self):
        """
        """
        a = ann.ANN([2,5,2], [af.sigmoid, af.softmax])
        inputs = np.array([[1, 4], [2, 3], [3, 1]])
        targets = np.array([[0.4, 0.6], [0.5, 0.5], [0.8, 0.2]])
        dw1 = a.gradient(inputs, targets, errorfunc=ef.entropic)
        dw2 = a._gradient_finite_difference(inputs, targets, errorfunc=ef.entropic)
        totsum = calc_weight_abs_diff(dw1, dw2)
        self.assertAlmostEqual(totsum, 0.0, places=3)
        
    def test_bruteforce_grad(self):
        """
        Test if the finite difference gradient works.
        """
        a = ann.ANN([2,2,3])
        a.weights = [np.array([[ 0.28012613, -2.44411668,  0.35384519],
                    [-0.9022525 , -0.49988908,  0.28618509]]),
                    np.array([[ 0.31478856, -0.21023631, -1.13039051],
                    [ 0.09771278, -0.40866703, -1.2686224 ],
                    [-0.89932226, -1.50500454, -2.41433566]])]
        dwt = [np.array([[  9.75499795e+00,   1.95099417e+00,   1.95099417e+00],
            [  1.56466164e-02,   3.12930948e-03,   3.12930948e-03]]),
            np.array([[ 3.0535242 ,  5.10754909, -5.10835203],
            [ 1.14674266,  1.91812621, -1.91842712],
            [ 3.211118  ,  5.37115205, -5.37199649]])]
        X = np.array([[5,1]])
        Y = np.array([[4,1,5]])
        dw = a._gradient_finite_difference(X, Y)
        totsum = calc_weight_abs_diff(dw, dwt)
        self.assertAlmostEqual(totsum, 0.0, places=4)
        
    def test_weight_copy(self):
        a = ann.ANN([5,5,5])
        w = a.get_weight_copy()
        totsum = calc_weight_abs_diff(w, a.weights)
        self.assertAlmostEqual(totsum, 0.0, places=8)

    def test_find_weight(self):
        """
        Checks find_weight and find_flat_weight_no
        """
        a = ann.ANN([5,7,3])
        N = np.size(a.get_flat_weights())
        for i in range(0, N):
            (layer, r, c) = a.find_weight(i)
            self.assertEqual(a.find_flat_weight_no(layer, r, c), i)

    def test_get_set_flat_weight(self):
        """
        Checks set_flat_weight and get_flat_weight
        """
        a = ann.ANN([5,7,3])
        N = np.size(a.get_flat_weights())
        for i in range(0, N):
            a.set_flat_weight(i, i)
            self.assertEqual(a.get_flat_weight(i), i)
            
    def test_set_flat_weight(self):
        """
        """
        a = ann.ANN([1,2,3])
        N = np.size(a.get_flat_weights())
        for i in range(0, N):
            a.set_flat_weight(i, i)
        target = [ np.array([[ 0.,  1.],
                             [ 2.,  3.]]),
                   np.array([[  4.,   5.,   6.],
                             [  7.,   8.,   9.],
                             [ 10.,  11.,  12.]])]
        self.assertEqual(True, np.all(a.weights[0] == target[0]))
        self.assertEqual(True, np.all(a.weights[1] == target[1]))

    def test_weigth_decay_gradient(self):
        """
        Test if the gradient is calculated correctly.
        """
        nn = ann.WeightDecayANN((2,4,2))
        x = np.array([[0,0],[0,1],[1,0],[1,1]])
        t = np.array([[0,1],[1,2],[1,3],[0,4]])
        # Gradient has been overwritten with a weight decay gradient
        dw1 = nn.gradient(x, t)
        # Here we get the error function with weight decay.
        dw2 = nn._gradient_finite_difference(x, t, nn.get_error_func())
        totsum = calc_weight_abs_diff(dw1, dw2)
        self.assertAlmostEqual(totsum, 0.0, places=3)
    # check get_flat_weights()
    # check set_flat_weights()
        

def calc_weight_abs_diff(w1, w2):
    """
    Returns the sum of the absolute difference between all the
    weights in w1 and w2. Used for comparing weight sets, and derived
    weights sets.
    """
    totsum = 0
    if (len(w1) != len(w2)):
        raise Exception('Lists should be the same size')
    for i in xrange(0, len(w1)):
        ww1 = w1[i]
        ww2 = w2[i]
        totsum += np.sum(np.abs(ww1-ww2))
    return totsum

    #TODO: check gradient, gradient_train, default af, setting af
if __name__ == '__main__':
    unittest.main()


