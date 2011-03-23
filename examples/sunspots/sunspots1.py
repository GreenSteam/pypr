#!/bin/python

# This code is based upon the exerice 4 from the IMM DTU course 02457
# written by Lars Kai Hansen & Karam Sidaros

from numpy import *
import pypr.preprocessing as preproc
import pypr.ann as ann
import pypr.optimization as opt
from pypr.helpers.modelwithdata import *
from pypr.stattest import *

# Load data
rg = genfromtxt('sp.dat')
year = rg[:,0].astype(int)
sp = rg[:,1]

d = 3 # Number of inputs
train_until_year = 1920 # including
last_train = nonzero(year==train_until_year)[0][0]-d

# Create lag space matrix 
N = len(year)-d
T = np.c_[sp[d:]]
X = ones((N,d))
for a in range(0, N):
  X[a,:] = sp[a:a+d]

# Training and test sets
Xtrain=X[0:last_train+1,:]
Ttrain=T[0:last_train+1,:]
Xtest=X[last_train+1:,:]
Ttest=T[last_train+1:,:]

# Normalize:
normX = preproc.Normalizer(X)
normT = preproc.Normalizer(T)
Xn = normX.transform(X)
Tn = normT.transform(T)

# Setup model:
nn = ann.WeightDecayANN([d, 4, 1])
nn.v = 0.0 # Weight decay, just a guess, should actually be found
dm = ModelWithData(nn, Xn, Tn)

# Train model:
err = opt.minimize(dm.get_parameters(), dm.err_func, dm.err_func_d, 200)

# Predict
Y = normT.invtransform(nn.forward(Xn))
SE = (Y-T)**2.0

# Plot data
figure()
subplot(211)
plot(year, rg[:,1], 'b', label='Target')
title('Yearly mean of group sunspot numbers')
xlabel('Year')
ylabel('Number')
plot(year[d:], Y, 'r', label='Output')
legend()
title('Sunspot predcition using a NN with %d inputs'%d)
axvline(train_until_year)
subplot(212)
plot(year[d:], SE)
xlabel('Year')
ylabel('$(y(x^n)-t^n)^2$')
title('Mean square error = %.2f'%np.mean(err))

# Plot sample autocorrelation of the residual of the fitted model
figure()
lags = range(0,21)
stem(lags, sac(Y-T, lags))
title('Sample Autocorrelation of residuals')
xlabel('Lag')
ylabel('Sample Autocorrelation')
