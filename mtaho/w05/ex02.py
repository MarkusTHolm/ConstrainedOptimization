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
from mtaho.src.Solvers import Solvers

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
                [0  ,1]])) 
A = At.T

b = - np.array([[2, 6, 2, 0, 0]]).T

### Manual active set algorithm:

## 1) We start at: x0 = (2, 0), i.e. W = {3, 5}

W = [2, 4]
Aw = A[:, W]
bw = b[W]

x, lam = Solvers.solveEqualityQP(H, g, Aw, bw, 'LDL')    

print(f"x = \n {x}")
print(f"lam = \n {lam}")

# x = [2, 0]
# lam = [-2, -1], so we drop c3

## 2) Then we solve the equality constrained subproblem for the direction
# min: 0.5*p'Hp + g'p 
# s.t.   Aw p = 0

xk = x
gk = H @ xk + g

W = [4]
Aw = A[:, W]
bw = b[W]

pk, lam = Solvers.solveEqualityQP(H, gk, Aw, np.zeros((len(W), 1)), 'LDL')    

print(f"pk = \n {pk}")
print(f"lam = \n {lam}")

# p = [-1, 0]
# lam = [-5]

# Find blocking constraints
nW = [0, 1, 2, 3]
Anw = A[:, nW]
bnw = b[nW]
print(f"Aw.T @ pk = \n {Anw.T @ pk}")

# Find alpha values from blocking constraints
iblock = [0, 3]
alphas = (b[iblock] - Anw[:, iblock].T @ xk)/( Anw[:, iblock].T @ pk )
alphak = np.min(np.minimum(1, alphas.flatten()))

# No blocking constraints (ai@pk > 0 for all i) so we step with alpha=1
x1 = x + alphak*pk

## 3) Solve new subproblem 

xk = x1
gk = H @ xk + g

W = [4]
Aw = A[:, W]
bw = b[W]

pk, lam = Solvers.solveEqualityQP(H, gk, Aw, np.zeros((len(W), 1)), 'LDL')    

print(f"pk = \n {pk}")
print(f"lam = \n {lam}")

# p = [0, 0]
# lam = [-5]

x2 = x1

# As p=0, we drop c5 as well and solve the unconstrained problem

## 4) Solve unconstrained problem from xk = [1, 0]

xk = x2
gk = H @ xk + g
pk = - np.linalg.solve(H, gk)

# Find blocking constraints
Anw = A
print(f"Aw.T @ pk =  \n {Anw.T @ pk}")

# Find step length
iblock = [0, 1]
alphas = (b[iblock] - Anw[:, iblock].T @ xk)/( Anw[:, iblock].T @ pk )
alphak = np.min(np.minimum(1, alphas))

# Step 
p2 = pk
x3 = x2 + alphak*pk

## 5) There is a single blocking constraint at this point c1, so W = {1}

xk = x3
gk = H @ xk + g

W = [0]
Aw = A[:, W]
bw = b[W]

pk, lam = Solvers.solveEqualityQP(H, gk, Aw, np.zeros((len(W), 1)), 'LDL')    

print(f"pk = \n {pk}")
print(f"lam = \n {lam}")

# Find blocking constraints
Anw = A
nW = [1, 2, 3, 4]
print(f"Aw.T @ pk =  \n {Anw.T @ pk}")

# Find step length
iblock = [1]
alphas = (b[iblock] - Anw[:, iblock].T @ xk)/( Anw[:, iblock].T @ pk )
alphak = np.min(np.minimum(1, alphas))

# Step 
x4 = x3 + alphak*pk

# No blocking constraints, so the next working set is unchanged

## 6) Find new step length: W = {1}

xk = x4
gk = H @ xk + g

W = [0]
Aw = A[:, W]
bw = b[W]

pk, lam = Solvers.solveEqualityQP(H, gk, Aw, np.zeros((len(W), 1)), 'LDL')    

print(f"pk = \n {pk}")
print(f"lam = \n {lam}")

# We have p=0 and the lam > 0, so we have found the solution!

x5 = x4
xs = x5

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
ax.plot(x1[0], x1[1], 'rs', label='$x^1$', markersize=ms*1.2)
ax.plot(x2[0], x2[1], 'go', label='$x^2$', markersize=ms)
ax.plot(x3[0], x3[1], 'bo', label='$x^3$', markersize=ms)
ax.plot(x4[0], x4[1], 'mo', label='$x^4$', markersize=ms)
ax.plot(xs[0], xs[1], 'o', color='tab:orange', label=r'$x^\ast$',
         markersize=ms*2, markerfacecolor=None)
# ax.plot(x2[0], x2[1], 'ro', label='$x^1$', markersize=ms)
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
fig.savefig(f'{workDir}/ex02.png')