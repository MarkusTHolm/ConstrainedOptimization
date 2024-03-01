import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import cvxpy as cp
import scipy
from scipy.optimize import minimize, Bounds
from scipy.optimize import LinearConstraint, NonlinearConstraint
import cyipopt
from cyipopt import minimize_ipopt

projectDir = "/home/mtaho/Code/Courses/ConstrainedOptimization"
sys.path.append(os.path.realpath(f"{projectDir}"))

from mtaho.plot_settings import define_plot_settings
from mtaho.src.FiniteDifference import FD
from mtaho.src.BasicOpt import Opt

workDir = f"{projectDir}/mtaho/w05"

def funContour(X, Y):
    tmp1 = X - 1
    tmp2 = Y - 2.5
    f = tmp1**2 + tmp2**2
    return f

x0 = np.array([2, 0])

H = np.array(([[2, 0],
               [0, 2]]))

g = np.array([[-2, -5]]).T

At = np.array(([[1  ,-2],
                [-1 ,-2],
                [-1 ,2],
                [1  ,0],
                [0  ,0]])) 
A = At.T






### Contour plot
xlim = [-1, 5]
ylim = [-1, 5]
fig, ax = Opt.contourPlot(funContour,
                          xlim=xlim, ylim=ylim,
                          colorScale='linear', figSize=(7, 5),
                          fontSize=18)
ms = 6

# Plot solutions
ax.plot(x0[0], x0[1], 'ko', label='$x^0$', markersize=ms)
# Draw constraints
xc = np.linspace(xlim[0], xlim[1])
yc = np.linspace(ylim[0], ylim[1])
alpha = 0.5
yc1 = (xc + 2)/2
yc2 = (-1*xc + 6)/2
yc3 = -(-1*xc + 2)/2
ax.fill_between(xc, yc1, np.repeat(5, 50), color='grey', alpha=alpha, label='$c_1$',
        hatch='\\\\') 
ax.fill_between(xc, yc2, np.repeat(5, 50), color='grey', alpha=alpha, label='$c_2$',
        hatch='**') 
ax.fill_between(xc, yc3, np.repeat(-5, 50), color='grey', alpha=alpha, label='$c_3$',
        hatch='///') 
ax.fill_between(xc, np.repeat(-5, 50), color='grey', alpha=alpha, label='$c_4$',
        hatch='xx') 
ax.fill_betweenx(yc, np.repeat(-5, 50), color='grey', alpha=alpha, label='$c_5$',
                hatch='++') 

ax.set_xlim(xlim)
ax.set_ylim(xlim)

ax.legend(bbox_to_anchor=(-0.16, 1))
ax.set_aspect('equal')
fig.tight_layout()
fig.savefig(f'{workDir}/ex02.png')