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
b = - np.array([[2.0, 6.0, 2.0, 0.0, 0.0]]).T

### Primal active set algorithm for convex QPs:

# 1) We start at: x0 = (2, 0), i.e. W = {3, 5}

x0 = np.array([[2, 0]]).T
W = [2, 4]

n = np.shape(x0)[0]
m = np.shape(A)[1]

maxiter = 10
numtol = 1e-6
lams = np.zeros((m, 1))
I = np.arange(m)

# Set initial vaues
xk = x0

for k in range(maxiter):

    # Define system from the working set W
    Aw = A[:, W]
    bw = b[W]   
    zerow = np.zeros((len(W), 1))

    # Solve the equality constrained QP for the search direction pk
    gk = H @ xk + g
    pk, lamk = Solvers.solveEqualityQP(H, gk, Aw, zerow, 'LDL')    

    print(f"Iteration: k = {k}")
    print(f"xk = \n {xk}")
    print(f"pk = \n {pk}")
    print(f"lamk = \n {lamk}")
    print(f"W = \n {W}")

    if np.isclose(np.linalg.norm(pk), 0):
    
        if np.all(lamk >= numtol):           
            # The optimal solution has been founds
            xs = xk            
            lams[W] = lamk
            break
        else:
            # Remove the most negative constraint from the working set
            j = np.argmin(lamk)
            W = np.setdiff1d(W, W[j])
            # xk = xk    
                
    else:    
        # Compute the distance to the nearest inactive constraint in the 
        # search direction pk

        # Extract system which is not contained in the working set
        nW = np.setdiff1d(I, W)             
        Anw = A[:, nW]
        bnw = b[nW]
        
        # Find blocking constraints
        blockCrit = Anw.T @ pk
        blockCrit = blockCrit.flatten() 
        blockCrit < 0
        iblock = nW[blockCrit < -numtol]

        # Find step length        
        alphas = (b[iblock] - Anw[:, iblock].T @ xk)/( Anw[:, iblock].T @ pk )
        alpha = np.minimum(1, alphas)
        j = np.argmin(alpha)
        alphak = alpha[j]

        if alphak < 1:
            xk = xk + alphak*pk
            W = np.append(W, j)
            W.sort()
        else:
            xk = xk + pk

    print("")

## Correct steps:
    
# k=0: xk = [2, 0]'  , pk = [0, 0]'     , lamk = [-2, -1]', W = {2, 4}

# k=1: xk = [2, 0]'  , pk = [-1, 0]'    , lamk = [-5]'    , W = {4}
    
# k=2: xk = [1, 0]'  , pk = [0, 0]'     , lamk = [-5]'    , W = {4}    
    
# k=3: xk = [1, 0]'  , pk = [0, 2.5]'   , lamk = []'      , W = {Ø}    
    
# k=4: xk = [1, 1.5]', pk = [0.4, 0.2]' , lamk = [..]'    , W = {1}    
    
# k=5: xk = [1.4, 1.7]', pk = [0, 0]'   , lamk = [0.8]'    , W = {1}    




print(f"Solution found after {k} iterations: x = \n {xs}")
















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