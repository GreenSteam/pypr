
# Plot activation functions

import numpy as np
from matplotlib.pylab import *
import pypr.ann.activation_functions as af

fig = figure()

x = linspace(-4, 4, 300)
y_tanh = np.tanh(x)
y_sigmoid = af.sigmoid[0](x)
y_squash = af.squash[0](x)

w = 2
line_lin = plot(x, x, '--k', label='lin', linewidth = w)
line_tanh = plot(x, y_tanh, 'k', label='tanh', linewidth = w)
line_sigmoid = plot(x, y_sigmoid, ':k', label='logistic sigmoid', linewidth = w)
line_squash = plot(x, y_squash, '-.k', label='squash', linewidth = w)

fig.gca().set_ylim(-2, 2)
fig.gca().set_xlim(-2, 2)
fig.gca().xaxis.grid(True)
fig.gca().yaxis.grid(True)
xlabel('x')
ylabel('g(x)')
title('Activation functions')
legend( (line_lin, line_tanh, line_sigmoid), ('linear', 'tanh', \
    'logistic sigmoid','squash'), loc=4)

#savefig('af.png')

