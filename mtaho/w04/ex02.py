import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import cvxpy as cp
import cvxopt 
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

workDir = f"{projectDir}/mtaho/w04"


def fun_contour(X, Y):
    return 3*X**2 + X*Y + 2*Y**2 + 3*X + 2*Y + 4

P = cvxopt.matrix([[6., 1.], 
                   [1., 4.]])
p = cvxopt.matrix([3., 2.])
G = -cvxopt.matrix([[1., 0.],
                   [0., 1.],
                   [1., 1.]]).T
h = -cvxopt.matrix([0., 0., 3.])

sol = cvxopt.solvers.qp(P, p, G, h)

xs = sol['x']

print(sol['x'])
print(sol['primal objective'])

xlim = [-5, 5]
ylim = [-5, 5]
fig, ax = Opt.contourPlot(fun_contour,
                          xlim=xlim, ylim=ylim, vmax=200,
                          colorScale='linear', figSize=(8, 5))
ms = 8

# Plot solutions
# ax.plot(x0[0], x0[1], 'ko', label='$x^0$', markersize=ms)
ax.plot(xs[0], xs[1], 'rs', markerfacecolor='none', markersize=ms, 
        label=r'$x^\ast_a$', markeredgewidth=2)
# ax.plot(res_b.x[0], res_b.x[1], 'rs', markerfacecolor='none', markersize=ms, 
#         label=r'$x^\ast_b$', markeredgewidth=2)

# Draw constraints
xc = np.linspace(xlim[0], xlim[1])
alpha = 0.5

yc1 = 0*xc
yc2 = 0*xc
yc3 = 3 - xc
ax.fill_betweenx(xc, yc1, np.repeat(-5, 50), color='grey', alpha=alpha, label='$c_1$',
        hatch='\\') 
ax.fill_between(xc, yc2, np.repeat(-5, 50), color='grey', alpha=alpha, label='$c_2$',
        hatch='*') 
ax.fill_between(xc, yc3, np.repeat(-5, 50), color='grey', alpha=alpha, label='$c_3$',
        hatch='//') 

ax.set_xlim(xlim)
ax.set_ylim(xlim)

ax.legend(bbox_to_anchor=(1.5, 1))
fig.tight_layout()
fig.savefig(f'{workDir}/ex02.png')