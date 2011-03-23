
# ---- Test ---------------------------------------------------------------------
#    def generate(self):
#        #n = 15
#        #x = 15*(np.random.rand(n,1)-0.5)
#        x = np.c_[np.array([-2.1775, -0.9235, 0.7502, -5.8868, -2.7995])]
#        rn = np.c_[np.array([ 1.4051, 1.1780, -1.1142, 0.2474, -0.8169])]
#        return np.dot(np.linalg.cholesky(self.cf.eval(x)), rn) #np.random.randn(n,1))
#    result should be: array([[ 1.41210802],
#       [ 1.69352704],
#       [-0.74440531],
#       [ 0.24932682],
#       [ 0.39784666]])
#a=cfSquaredExponentialIso()
#b=cfNoise(np.log(0.1))
#c=a+b


import unittest
import copy
from GaussianProcess import *

class TestSequenceFunctions(unittest.TestCase):

    def setUp(self):
        self.a = cfSquaredExponentialIso()
        self.b = cfNoise(np.log(0.1))
        self.c = self.a + self.b
        self.g = GaussianProcess(self.c)

    def test_GPR_partial_derivatives(self):
        """
        """
        D = 50 # 5-d input
        cf = cfSquaredExponentialARD(np.random.randn(D).tolist()) + cfNoise()
        g = GaussianProcess(cf)
        X = np.random.rand(200, D) 
        y = np.c_[np.random.rand(200)]
        (Lorig, der) = g.find_likelihood_der(X, y) # Find likelihood and its derivatives
        d = 10**-6 # Finite difference to use
        # Find the partial derivatives
        par = g.cf.get_params()
        der_fd = np.zeros(len(par))
        for i in range(len(par)):
            par_d = copy.copy(par)
            par_d[i] = par_d[i] + d
            g.cf.set_params(par_d)
            (L_d, dummy) = g.find_likelihood_der(X, y)
            der_fd[i] = (L_d - Lorig) / d
        self.assertAlmostEqual(np.sum(der-der_fd), 0.0, places=2)

    def test_generate(self):
        """
        """
        # make sure that the output is correct
        x = np.c_[np.array([-2.1775, -0.9235, 0.7502, -5.8868, -2.7995])]
        rn = np.c_[np.array([ 1.4051, 1.1780, -1.1142, 0.2474, -0.8169])]
        t = np.array([[ 1.41210802], [ 1.69352704], [-0.74440531], [ 0.24932682], [ 0.39784666]])
        y = self.g.generate(x, rn=rn)
        s = np.sum(t-y)
        self.assertAlmostEqual(s, 0.0)
        
    def test_nllikeliness_and_der(self):
        x = np.c_[np.array([1,2,3])]
        y = x
        nll, ll_der = self.g.find_likelihood_der(x, y)
        ll_derT = np.array([-1.94060801, -6.27147378, -0.03801839])
        self.assertAlmostEqual(nll, 6.9120784760861937)
        self.assertAlmostEqual(np.sum(ll_der-ll_derT), 0)

    def test_init_params(self):
        a = cfSquaredExponentialIso(1,2)
        b = cfNoise(3)
        c = cfSquaredExponentialARD([4, 5, 6, 7, 8], 9)
        d = (a + b) + c
        self.assertEqual((a.get_params()+b.get_params())+c.get_params(), \
                         [1, 2, 3, 4, 5, 6, 7, 8, 9])

    def test_set_params(self):
        a = cfSquaredExponentialIso(1, 2)
        b = cfNoise(3)
        c = cfSquaredExponentialARD([4, 5, 6, 7, 8], 9)
        d = (a + b) + c
        d.set_params([20, 21, 22, 23, 24, 25, 26, 27, 28])
        self.assertEqual(a.get_params(), [20, 21])
        self.assertEqual(b.get_params(), [22])
        self.assertEqual(c.get_params(), [23, 24, 25, 26, 27, 28])
        self.assertEqual(d.get_params(), [20, 21, 22, 23, 24, 25, 26, 27, 28])

    def test_sq_dist(self):
        X = np.array([[1, 2], [3, 4]])
        T1 = np.array([[ 0.,  8.], [ 8.,  0.]])
        self.assertAlmostEqual(np.sum(sq_dist(X)-T1), 0)
        T2 = np.array([[ 1.,  5.], [ 5.,  1.]])
        self.assertAlmostEqual(np.sum(sq_dist(X,X.T)-T2), 0)
        
    def test_cfSquaredExponentialIso_eval(self):
        x = np.c_[np.array([1,2])]
        x2 = np.c_[np.array([1,2,3])]
        y, s2 = self.a.eval(x,x2)
        yT = np.array([[ 1.],[ 1.],[ 1.]])
        s2T = np.array([[ 1., 0.60653066, 0.13533528],
                        [ 0.60653066, 1., 0.60653066]])
        self.assertAlmostEqual(np.sum(y-yT), 0)
        self.assertAlmostEqual(np.sum(s2-s2T), 0)
        
    #TODO: test cfNoise_eval
 
    def test_cfSquaredExponential_derivative(self):
        X = np.c_[np.array([1,2])]
        dr0 = self.a.derivative(X, 0)
        dr1 = self.a.derivative(X, 1)
        T0 = np.array([[ 0., 0.60653066],
                       [ 0.60653066, 0.]])
        T1 = np.array([[ 2., 1.21306132],
                       [ 1.21306132, 2.]])
        self.assertAlmostEqual(np.sum(dr0-T0), 0)
        self.assertAlmostEqual(np.sum(dr1-T1), 0)
    
    def test_cfNoise_derivative(self):
        X = np.c_[np.array([1,2,3,4])]
        dr0 = self.b.derivative(X, 0)
        self.assertAlmostEqual(np.sum(dr0-0.02*np.eye(len(X))), 0)
        
    def test_sum_derivative(self):
        X = np.c_[np.array([1,2])]
        r = self.a.derivative(X, 0) + self.b.derivative(X, 0)
        t = np.array([[ 0.02, 0.60653066],
                      [ 0.60653066, 0.02]])
        self.assertAlmostEqual(np.sum(r-t), 0)

    def test_regression(self):
        X = np.c_[np.array([1,2,3,4])]
        y = np.c_[np.array([0,2,1,-3])]
        xs = np.c_[np.linspace(0,4,11)]
        g = GaussianProcess(self.c)
        yy, s2 = g.regression(X, y, xs)
        yyT = np.array([[-0.50362336],
                   [-0.51735067],
                   [-0.25343978],
                   [ 0.36155214],
                   [ 1.21815483],
                   [ 1.98742161],
                   [ 2.22354088],
                   [ 1.60100307],
                   [ 0.16916126],
                   [-1.57884763],
                   [-2.94742789]])
        s2T = np.array([[ 0.5333147 ],
                        [ 0.21704342],
                        [ 0.03898559],
                        [ 0.02492656],
                        [ 0.03084483],
                        [ 0.01967766],
                        [ 0.02672439],
                        [ 0.02225341],
                        [ 0.02372749],
                        [ 0.03226112],
                        [ 0.01981508]])
        self.assertAlmostEqual(np.sum(s2-s2T), 0)
        self.assertAlmostEqual(np.sum(yy-yyT), 0)

    def test_cfSquaredExponentialARD_der(self):
        d = cfSquaredExponentialARD()
        T0 = np.array([[ 0., 0.60653066, 0.54134113],
                       [ 0.60653066, 0.,  0.60653066],
                       [ 0.54134113, 0.60653066, 0.]])
        T1 = np.array([[ 2., 1.21306132, 0.27067057],
                       [ 1.21306132, 2., 1.21306132],
                       [ 0.27067057, 1.21306132, 2.]])
        X=np.c_[[1,2,3]]
        self.assertAlmostEqual(np.sum(d.derivative(X,0)-T0), 0)
        self.assertAlmostEqual(np.sum(d.derivative(X,1)-T1), 0)
        
    def test_cfSquaredExponentialArd_eval(self):
        d = cfSquaredExponentialARD()
        T = np.array([[ 1., 0.60653066, 0.13533528],
                      [ 0.60653066, 1., 0.60653066],
                      [ 0.13533528, 0.60653066, 1.]])
        X=np.c_[[1,2,3]]
        self.assertAlmostEqual(np.sum(d.eval(X)-T), 0)
        y=X
        T1 = np.array([[ 1.],[ 1.],[ 1.]])
        T2 = np. array([[ 1., 0.60653066, 0.13533528],
                        [ 0.60653066, 1., 0.60653066],
                        [ 0.13533528, 0.60653066, 1.]])
        r = d.eval(X,y)
        self.assertAlmostEqual(np.sum(r[0]-T1), 0)
        self.assertAlmostEqual(np.sum(r[1]-T2), 0)

    def test_optimization(self):
        # Also make sure that the example works
        x = np.c_[[-6.9292, -1.4709, 0.4831, 3.0603, -3.3495, -4.6909, 1.9009, 1.5049,
                5.0070, 0.5561, 0.3776, 2.5393, -5.5687, -0.1464, -4.1271, -0.6450,
                -5.4380, -4.2863, 0.5167, -2.5414]]
        y = np.c_[[0.8157, 0.2669, 0.0189, -0.3197, 0.5200, 1.2551, -0.8227, -0.4394,
                0.1169, 0.0805, 0.0551, -0.6780, 1.0632, 0.0647, 1.0685, 0.0094, 
                1.2311, 1.1087, 0.0296, 0.6057]] 
        sq_exp = cfSquaredExponentialIso(2, 2)
        noise = cfNoise(2)
        jitter = cfJitter()
        cov_func = sq_exp + noise #+ jitter
        g = GaussianProcess(cov_func)
        gpr = GPR(g, x, y, mean_tol=2)
        diff = np.exp(g.cf.get_params()) - \
            np.array([1.12077449, 0.59151511, 0.06164603])
        self.assertAlmostEqual(np.sum(diff), 0)

if __name__ == '__main__':
    unittest.main()


