import matplotlib.pyplot as plt
import numpy as np
import os
import sys

projectDir = "/home/mtaho/Code/Courses/ConstrainedOptimization"
sys.path.append(os.path.realpath(f"{projectDir}"))

from mtaho.plot_settings import define_plot_settings
from mtaho.src.FiniteDifference import FD
from mtaho.src.BasicOpt import Opt

workDir = f"{projectDir}/mtaho/w02"


def funContour(X, Y):
    f = X**2 + Y**2 + 3*Y
    return f

xlim = [-5, 5]
ylim = [-5, 5]
fig, ax = Opt.contourPlot(funContour,
                          xlim=xlim, ylim=ylim, figSize=(8, 5))

alpha = 1 #0.75
nxy = 10000
xx = np.linspace(xlim[0], xlim[1], nxy)
yy = np.linspace(ylim[0], ylim[1], nxy)

yc1 = -1 - np.sqrt(-xx**2 + 1)
yc2 = -1 + np.sqrt(-xx**2 + 1)

ax.plot(xx, yc1, 'r-', label='$c_1$') 
ax.plot(xx, yc2, 'r-') 
# ax.plot(3, 0, 'ro', label='$x^\\ast$')

# ax.set_xlim(xlim)
# ax.set_ylim(xlim)

ax.legend(bbox_to_anchor=(1.5, 1))
fig.tight_layout()
fig.savefig(f'{workDir}/ex3Contour.png')


