import matplotlib.pyplot as plt
import numpy as np
import os
import sys

# from intvalpy import lineqs

projectDir = "/home/mtaho/Code/Courses/ConstrainedOptimization"
sys.path.append(os.path.realpath(f"{projectDir}"))

from mtaho.plot_settings import define_plot_settings
from mtaho.src.FiniteDifference import FD
from mtaho.src.BasicOpt import Opt

workDir = f"{projectDir}/mtaho/w02"


def funContour(X, Y):
    g = np.array([[1, -2]]).T
    f = g[0]*X + g[1]*Y
    return f

A = np.array([[1, 0,  1,  1, -5],
              [0, 1, -1, -5,  1]])
b = np.array([0, 0, -2, -20, -15])

xlim = [-5, 5]
ylim = [-5, 5]

# Solution
xs = [5/2, 9/2]

fig, ax = Opt.contourPlot(funContour,
                          xlim=xlim, ylim=ylim, figSize=(8, 5))

alpha = 1 #0.75
nxy = 50
xx = np.linspace(xlim[0], xlim[1], nxy)
yy = np.linspace(ylim[0], ylim[1], nxy)

xc1 = np.repeat(0, nxy)
yc2 = np.repeat(0, nxy)
yc3 = -(A[0, 2]*xx - b[2])/A[1, 2]
yc4 = -(A[0, 3]*xx - b[3])/A[1, 3]
yc5 = -(A[0, 4]*xx - b[4])/A[1, 4]

ax.fill_betweenx(yy, xc1, np.repeat(xlim[0], nxy),
                alpha=alpha, label='$c_1$', hatch='\\') 
ax.fill_between(xx, yc2, np.repeat(ylim[0], nxy),
                alpha=alpha, label='$c_2$', hatch='/') 
ax.fill_between(xx, yc3, np.repeat(ylim[1], nxy),
                alpha=alpha, label='$c_3$', hatch='o') 
ax.fill_between(xx, yc4, np.repeat(ylim[1], nxy),
                alpha=alpha, label='$c_4$', hatch='*') 
ax.fill_between(xx, yc5, np.repeat(ylim[0], nxy),
                alpha=alpha, label='$c_5$', hatch='/') 
ax.plot(xs[0], xs[1], 'bo', label='$x^\\ast$', markersize=10)

ax.set_xlim(xlim)
ax.set_ylim(xlim)

ax.legend(bbox_to_anchor=(1.6, 1))
fig.tight_layout()
fig.savefig(f'{workDir}/ex2Contour.png')



# lineqs(-A.T, -b, title='Solution', color='gray', alpha=0.5, s=10, size=(15,15), save=True, show=True)
# plt.show()