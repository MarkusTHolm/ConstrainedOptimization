import matplotlib.pyplot as plt
from cycler import cycler
import numpy as np
import os
import sys
import cvxpy as cp
import cvxopt 
import scipy as sp
from scipy.optimize import minimize, Bounds
from scipy.optimize import LinearConstraint, NonlinearConstraint

projectDir = "/home/mtaho/Code/Courses/ConstrainedOptimization"
# projectDir = "C:/Users/marku/Programming/ConstrainedOptimization"
sys.path.append(os.path.realpath(f"{projectDir}"))

from mtaho.plot_settings import define_plot_settings
from mtaho.src.FiniteDifference import FD
from mtaho.src.BasicOpt import Opt
from mtaho.src.Solvers import Solvers

workDir = f"{projectDir}/mtaho/w09/"

def contourFunHimmelblau(X, Y):
    tmp1 = X**2 + Y - 11
    tmp2 = X + Y**2 - 7
    f = tmp1**2 + tmp2**2
    return f

## Objective function and gradients
def funJacHess(x):
    x = x[:, 0]
    tmp1 = x[0]**2 + x[1] - 11
    tmp2 = x[0] + x[1]**2 - 7
    f = np.array([[tmp1**2 + tmp2**2]],
                 dtype=np.float64)
    df = np.array([[4*(x[0]**2 + x[1] - 11)*x[0] + 2*(x[0] + x[1]**2 - 7),
                   2*(x[0]**2 + x[1] - 11) + 4*(x[0] + x[1]**2 - 7)*x[1]]],
                   dtype=np.float64).T
    d2f = np.array([[12*x[0]**2 + 4*x[1] - 42 ,4*x[0] + 4*x[1]         ],
                    [4*x[0] + 4*x[1]          ,12*x[1]**2 + 4*x[0] - 26]],
                    dtype=np.float64)
    return f, df, d2f

def consJacHess(x):
    x = x[:, 0]
    c = np.array([[(x[0] + 2)**2 - x[1]]],
                 dtype=np.float64)
    dc = np.array([[2*(x[0] + 2), -1]],
                  dtype=np.float64).T
    d2c = np.zeros((2, 2, 1),
                   dtype=np.float64)
    d2c[:, :, 0] = np.array([[2, 0], [0, 0]])
    return c, dc, d2c

# Starting point
x0 = np.array([[-3, -1]], dtype=np.float64).T

# Evaluate functions
f, df, d2f = funJacHess(x0)
c, dc, d2c = consJacHess(x0)

sol = Solvers.SQPSolver(funJacHess, x0, EQConsFun=consJacHess)
solBFGS = Solvers.SQPSolver(funJacHess, x0, EQConsFun=consJacHess, 
                            BFGS=True, lineSearch=True)
solLine = Solvers.SQPSolver(funJacHess, x0, EQConsFun=consJacHess,
                            lineSearch=True)
solBFGSLine = Solvers.SQPSolver(funJacHess, x0, EQConsFun=consJacHess,
                            BFGS=True, lineSearch=True)

# Plot solutions on contour plot:
xlim = [-5, 5]
ylim = [-5, 5]
fig, ax = Opt.contourPlot(contourFunHimmelblau,
                          xlim=xlim, ylim=ylim, vmax=200,
                          colorScale='linear', figSize=(12, 6))
ms = 8
ax.plot(x0[0], x0[1], 'ko', label='$x^0$', markersize=ms)
def plotIterationArrows(xsol, color, label):
    for k in range(1, np.shape(xsol)[1]-1):
        x0 = xsol[:, k-1]
        x1 = xsol[:, k]
        dx0 = x1 - x0
        ax.arrow(x0[0], x0[1], dx0[0], dx0[1], color=color,
                head_width=0.15, length_includes_head=True)
    ax.plot(0, 0, color, label=label)

plotIterationArrows(sol['xStore'], 'r', 'Analytical Hessian')
plotIterationArrows(solBFGS['xStore'], 'b', 'Modified BFGS')
plotIterationArrows(solLine['xStore'], 'lime', 'Analytical Hessian w. LS')
plotIterationArrows(solBFGSLine['xStore'], 'magenta', 'Modified BFGS w. LS ')

# Draw constraints
xc = np.linspace(xlim[0], xlim[1])
yc1 = (xc+2)**2
ax.plot(xc, yc1, 'k', label='$c_1(x)$')
ax.set_xlim(xlim)
ax.set_ylim(xlim)
plt.legend(bbox_to_anchor=(1.2, 1), loc="upper left")
fig.tight_layout()
fig.savefig(f'{workDir}/ex01Contour.png')