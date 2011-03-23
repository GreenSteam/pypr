from pypr.stattest.ljungbox import *
import scipy.stats

x = np.random.randn(100)
#rg = genfromtxt('sunspots/sp.dat')
#x = rg[:,1] # Just use number of sun spots, ignore year
h = 20 # Number of lags
lags = range(h)
sa = np.zeros((h))
for k in range(len(lags)):
    sa[k] = sac(x, k)
figure()
markerline, stemlines, baseline = stem(lags, sa)
grid()
title('Sample Autocorrealtion Function (ACF)')
ylabel('Sample Autocorrelation')
xlabel('Lag')
h, pV, Q, cV = lbqtest(x, range(1, 20), alpha=0.1)
print 'lag   p-value          Q    c-value   rejectH0'
for i in range(len(h)):
    print "%-2d %10.3f %10.3f %10.3f      %s" % (i+1, pV[i], Q[i], cV[i], str(h[i]))
