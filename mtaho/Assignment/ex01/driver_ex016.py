import os
os.environ["OMP_NUM_THREADS"] = "1" # LIMIT NUMBER OF THREADS FOR MEASURING

import matplotlib.pyplot as plt
import numpy as np
import sys
import cvxpy as cp
import cvxopt 
import scipy as sp
from scipy.optimize import minimize, Bounds
from scipy.optimize import LinearConstraint, NonlinearConstraint
import datetime
import logging
import timeit
import pandas as pd
# import cyipopt
# from cyipopt import minimize_ipopt

projectDir = "/home/mtaho/Code/Courses/ConstrainedOptimization"
# projectDir = "C:/Users/marku/Programming/ConstrainedOptimization"
sys.path.append(os.path.realpath(f"{projectDir}"))

from mtaho.plot_settings import define_plot_settings
from mtaho.src.FiniteDifference import FD
from mtaho.src.BasicOpt import Opt
from mtaho.src.Solvers import Solvers

workDir = f"{projectDir}/mtaho/Assignment/ex01"

### 1) Define problem
H = np.array([[5.0000, 1.8600, 1.2400, 1.4800, -0.4600],
              [1.8600, 3.0000, 0.4400, 1.1200, 0.5200],
              [1.2400, 0.4400, 3.8000, 1.5600, -0.5400],
              [1.4800, 1.1200, 1.5600, 7.2000, -1.1200],
              [-0.4600, 0.5200, -0.5400, -1.1200, 7.8000]])
g = np.array([[-16.1000,-8.5000,-15.7000,-10.0200,-18.6800]]).T
A = np.array([[16.1000, 1.0000],
              [8.5000, 1.0000],
              [15.7000, 1.0000],
              [10.0200, 1.0000],
              [18.6800, 1.0000]])
b = np.array([[15.0, 1.0]]).T

### 2) Solve KKT system using LU factorization
x0, lam0 = Solvers.solveEqualityQP(H, g, A, b, type='LDL')
r0 = np.vstack((g, b)) 
print("x1 = \n", x0)
print("lam1 = \n", lam0)

# Brute force:
n, m = np.shape(A)
N = 10
b1Array = np.linspace(8.5, 18.68, N)
x1Array = np.zeros((n, N))
for i, b1 in enumerate(b1Array):
    b[0] = b1
    x1, lam1 = Solvers.solveEqualityQP(H, g, A, b, type='LDL')
    x1Array[:, i:i+1] = x1

# Using sensitivies:
K, r, m = Solvers.EqualityQPKKT(H, g, A, b, sparse=False)
dKdr = -np.linalg.inv(K)
x2Array = np.zeros((n, N))
for i, b1 in enumerate(b1Array):
    b[0] = b1
    r = np.vstack((g, b))
    z = dKdr@(r-r0)
    x2 = x0 + z[0:n]
    x2Array[:, i:i+1] = x2


define_plot_settings(20)
fig, ax = plt.subplots(figsize=(14, 4))
lw = 2
for i in range(n):
    plt.plot(b1Array, x1Array[i, :], '-', label=f"$x_{i+1}$ - Sensitivity",
             linewidth=lw)
plt.gca().set_prop_cycle(None)
for i in range(n):
    plt.plot(b1Array, x2Array[i, :], 's', label=f"$x_{i+1}$ - Brute force", 
             markerfacecolor=None, markeredgewidth=2, markersize=6,
             linewidth=lw)

ax.legend(bbox_to_anchor=(1, 1), loc="upper left", ncols=2)
ax.set_xlabel("$b_1$")
ax.set_ylabel("$x_i$")
fig.tight_layout()
# plt.show()
plt.savefig(f"{workDir}/TestProblem2Solution.png", dpi=600)