import matplotlib.pylab as pl

def plot_gpr(xs, ys, s2, axes=None,
        line_settings={'color':'black', 'lw':2.0}, 
        shade_settings={'facecolor':'lightyellow', 'alpha':0.5}):
    """
    Plot the mean predicted values and 95% confidence interval, two times
    the standard error, as a shaded area.

    Parameters
    ----------
    xs: array
        an N length np array of the x* data
    ys: array
        an N length np array of the y* data
    s2: array
        an N length np array of the variance, var(y*), data
    line_setting: dictionary, optional
        An dictionary with keywords and values to pass to the mean line of the
        plot. For example {'linewidth':2}
    shade_setting: dictionary, optional
        An dictionary with keywords and values to pass to the fillbetween
        function.
    axes: axes, optional
        The axes to put the plot.
        If axes is not specified then the current will be used (in pylab).

    Returns
    -------
    Returns a tuple with
    line: matplotlib.lines.Line2D
        the mean line
    poly: matplotlib.collections.PolyCollection
        shaded 95% confidence area
    """
    ax = axes
    if ax==None:
        ax = pl.gca()
    xsf = xs.flatten()
    ysf = ys.flatten()
    s2f = 2*pl.sqrt(s2.flatten())
    verts = zip(xsf, ysf+s2f) + zip(xsf[::-1], (ysf-s2f)[::-1])
    poly = ax.fill_between(xsf, ysf-s2f, ysf+s2f, **shade_settings)
    line = ax.plot(xs, ys, **line_settings)
    return line, poly


