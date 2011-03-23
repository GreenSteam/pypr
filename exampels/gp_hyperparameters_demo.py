
# This is based upon the slider_demo from matplotlib

from pylab import *
from matplotlib.widgets import Slider, Button, RadioButtons
import pypr.gp as gp
import pypr.gp.plot_gp as plot_gp

se_iso = gp.cfSquaredExponentialARD([log(1)], log(1))
noise = gp.cfNoise(log(0.1))
cov_func = se_iso + noise
g = gp.GaussianProcess(cov_func)
orig_values = cov_func.get_params()

# Sample some random points from the Gaussian Process
N = 25
delta = 15
x = delta*(rand(N,1)-0.5)
y = g.generate(x)

ax = subplot(111)
subplots_adjust(bottom=0.30, top=0.85)
xs = c_[linspace(-delta, delta, 200)]
ys, s2 = g.regression(x, y, xs)

# Main plot:
axis([-delta/1.5, delta/1.5, -2.0, 2.0])

def update(val):
    ax.cla()
    ax.set_title(r'GP samples with $\mathrm{log}(\mathit{l})='+'%.2f'%orig_values[0]+'$, '
                '$\mathrm{log}(\sigma_f)=' + '%.2f$'%orig_values[1] +
                ', and $\mathrm{log}(\sigma_n)=' + '%.2f$'%orig_values[2])
    l, = ax.plot(x, y, 'x')
    se_iso.set_params([s_sls.val, s_sm.val])
    print "update: ", s_nm.val
    noise.set_params([s_nm.val])
    ys, s2 = g.regression(x, y, xs)
    plot_gp.plot_gpr(xs, ys, s2, axes=ax)
    (nll, ders) = g.find_likelihood_der(x, y)
    ax.text(0.95, 0.05, '-log(Likelihood) = %.2f' % nll, transform=ax.transAxes, fontsize=10,
        verticalalignment='bottom', horizontalalignment='right')
    draw()

# Sliders:
axcolor = 'lightgoldenrodyellow'
ax_sls = axes([0.40, 0.20, 0.50, 0.03], axisbg=axcolor)
ax_sm  = axes([0.40, 0.15, 0.50, 0.03], axisbg=axcolor)
ax_nm  = axes([0.40, 0.1, 0.50, 0.03], axisbg=axcolor)
s_sls = Slider(ax_sls, r'Log char. length-scale, $\mathrm{log}(\mathit{l})$', -4.0, 4.0, valinit=se_iso.get_params()[0])
s_sm = Slider(ax_sm, r'Log ignal magnitude, $\mathrm{log}(\sigma_f)$', -4.0, 4.0, valinit=se_iso.get_params()[1])
s_nm = Slider(ax_nm, r'Log noise magnitude, $\mathrm{log}(\sigma_n)$', -4.0, 4.0, valinit=noise.get_params()[0])
s_sls.on_changed(update)
s_sm.on_changed(update)
s_nm.on_changed(update)

optimize_ax = axes([0.8, 0.025, 0.1, 0.04])
button = Button(optimize_ax, 'Optimize', color=axcolor, hovercolor='0.975')
def reset(event):
    gpr = gp.GPR(g, x, y, mean_tol=inf)
    hp = gpr.GP.cf.get_params()
    s_sls.set_val(hp[0])
    s_sm.set_val(hp[1])
    s_nm.set_val(hp[2])
    update(1)
button.on_clicked(reset)

update(1)
show()


