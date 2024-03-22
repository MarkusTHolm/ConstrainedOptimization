import matplotlib.pyplot as plt
from cycler import cycler
import numpy as np
import os
import sys
import cvxpy as cp
import cvxopt 
import scipy
from scipy.optimize import minimize, Bounds
from scipy.optimize import LinearConstraint, NonlinearConstraint
# import cyipopt
# from cyipopt import minimize_ipopt

projectDir = "/home/mtaho/Code/Courses/ConstrainedOptimization"
sys.path.append(os.path.realpath(f"{projectDir}"))

from mtaho.plot_settings import define_plot_settings
from mtaho.src.FiniteDifference import FD
from mtaho.src.BasicOpt import Opt
from mtaho.src.Solvers import Solvers

workDir = f"{projectDir}/mtaho/w05"

def funContour(X, Y):
    tmp1 = X - 1
    tmp2 = Y - 2.5
    f = tmp1**2 + tmp2**2
    return f

## Problem
# min_x : 0.5*x'Hx + g'x
# s.t.  : A'x >= b

H = np.array(([[2, 0],
               [0, 2]]))
g = np.array([[-2, -5]]).T
At = np.array(([[1.0  ,-2.0],
                [-1.0 ,-2.0],
                [-1.0 ,2.0],
                [1.0  ,0.0],
                [0.0  ,1.0]])) 
A = At.T
b = np.array([[2.0, 6.0, 2.0, 0.0, 0.0]]).T

### Primal active set algorithm for convex QPs:

## Correct steps (Nocedal & Wright):
    
# k=0: xk = [2, 0]'  , pk = [0, 0]'     , lamk = [-2, -1]', W = {2, 4}

# k=1: xk = [2, 0]'  , pk = [-1, 0]'    , lamk = [-5]'    , W = {4}
    
# k=2: xk = [1, 0]'  , pk = [0, 0]'     , lamk = [-5]'    , W = {4}    
    
# k=3: xk = [1, 0]'  , pk = [0, 2.5]'   , lamk = []'      , W = Ø    
    
# k=4: xk = [1, 1.5]', pk = [0.4, 0.2]' , lamk = [..]'    , W = {0}    
    
# k=5: xk = [1.4, 1.7]', pk = [0, 0]'   , lamk = [0.8]'   , W = {0}    


## Test method
x0 = np.array([[2, 0]]).T
W = [2, 4]
sol = Solvers.QPSolverInequalityActiveSet(H, g, A, b, x0, W=W)

print(sol)

### Contour plot
xlim = [-1, 5]
ylim = [-1, 5]
fig, ax = Opt.contourPlot(funContour,
                          xlim=xlim, ylim=ylim,
                          colorScale='linear', figSize=(7, 5),
                          fontSize=18)
ms = 6

custom_cycler = (cycler(color=['r', 'b', 'g', 'm', 'c', 'y']))
ax.set_prop_cycle(custom_cycler)

# Plot solutions
ax.plot(x0[0], x0[1], 'ko', label='$x^0$', markersize=ms)
for i, x in enumerate(sol['xiter'].T):
    ms = 6
    mfc = None
    if i % 2:
        ms = 8
        mfc = 'none'    
    ax.plot(x[0], x[1], 'o', label=f'$x^{i}$', markersize=ms,
            markerfacecolor=mfc, markeredgewidth=2)


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
        hatch='o') 
ax.fill_betweenx(yc, np.repeat(-5, 50), color='grey', alpha=alpha, label='$c_4$',
                hatch='++') 
ax.fill_between(xc, np.repeat(-5, 50), color='grey', alpha=alpha, label='$c_5$',
        hatch='xx') 

ax.set_xlim(xlim)
ax.set_ylim(xlim)

ax.legend(bbox_to_anchor=(-0.16, 1))
ax.set_aspect('equal')
fig.tight_layout()
fig.savefig(f'{workDir}/ex029.png')