#!/usr/bin/env python
from numpy import linspace,exp
from numpy.random import randn
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline

########## Univariate Fit
x = linspace(-5, 5, 200)
y = exp(-x**2) + randn(200)/10
s = UnivariateSpline(x, y, s=1)
xs = linspace(-5, 5, 1000)
ys = s(xs)
plt.plot(x, y, '.-')
plt.plot(xs, ys)
plt.show()
