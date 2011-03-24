
import numpy as np
import pypr.preprocessing as preproc
import pypr.ann as ann
import pypr.optimization as opt
from pypr.helpers.wrappers import *
from pypr.helpers.modelwithdata import *

D = np.loadtxt('data/shipfuel.csv.gz', skiprows=1, delimiter=',')
names = {'waterspeed':0, 'fuel':1, 'trim':2, 'windspeed':3, 'windangle':4,
        'pitch':5, 'portrudder':6, 'starboardrudder':7, 'heeling':8,
        'draft':9 }

targetCol = [1]; inputCol = [0, 2, 3, 4, 5, 6, 7, 8, 9]
T = D[:, targetCol]
X = D[:, inputCol]

# Normalize:
normX = preproc.Normalizer(X)
normT = preproc.Normalizer(T)
Xn = normX.transform(X)
Tn = normT.transform(T)

# Setup model:
nn = ann.WeightDecayANN([len(inputCol), 2, 1])
nn.v = 0.1 # Weight decay, just a guess, should actually be found
dm = ModelWithData(nn, Xn, Tn)

# Train model:
err = opt.minimize(dm.get_parameters(), dm.err_func, dm.err_func_d, 10)

# Wrap model for easy use:
ev = wrap_model(X, (normX.transform, nn.forward, normT.invtransform), **names)

print "ev(X) =", ev(X)
print "ev(waterspeed=20) =", ev(waterspeed=20)
print "ev(X, waterspeed=20) =", ev(X, waterspeed=20)
