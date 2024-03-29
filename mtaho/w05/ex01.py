import os
os.environ["OMP_NUM_THREADS"] = "1" # LIMIT NUMBER OF THREADS FOR MEASURING

import matplotlib.pyplot as plt
import numpy as np
import sys
import cvxpy as cp
import cvxopt 
import scipy
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

workDir = f"{projectDir}/mtaho/w05"

### 3) Define problem

def setupProblem(N, uBar, d0):
    n = N + 1                           # No. of design variables
    m = N                               # No. of constraints
    H = np.identity(n)
    g = -uBar*np.ones((n, 1))
    
    # Find transpose of At = A^T
    
    # First constraint
    At = np.zeros((m, n))
    At[0, 0] = -1
    At[0, n-2] = 1
    
    # Middle constraints
    for j in range(1, n-2):
        At[j, j-1] = 1
        At[j, j] = -1 
    
    # Last constraint
    At[-1, -3] = 1
    At[-1, -2] = -1
    At[-1, -1] = -1 

    # Find A - without transpsoe
    A = At.transpose()

    # Right-hand side of constraints
    b = np.zeros((m, 1))
    b[0] = -d0

    return H, g, A, b

# Problem parameters
uBar = 0.2
d0 = 0.1
N = 4

n = N + 1                           # No. of design variables
m = N                               # No. of constraints

H, g, A, b = setupProblem(N, uBar, d0)

print("H = \n", H)
print("g = \n", g)
print("At = \n", A.transpose())
print("b = \n", b)

### 4) Setup KKT system
K, r, m = Solvers.EqualityQPKKT(H, g, A, b)
print("K = \n", K)
print("r = \n", r)

### 5) Solve KKT system using LU factorization
x1, lam1 = Solvers.EqualityQPSolverLU(K, r, m)
print("x1 = \n", x1)
print("lam1 = \n", lam1)

### 6) Solve KKT system using LDL factorization
x2, lam2 = Solvers.EqualityQPSolverLDL(K, r, m)
print("x2 = \n", x2)
print("lam2 = \n", lam2)

### 6) Solve KKT system using Null-Space procedure
x3, lam3 = Solvers.EqualityQPSolverNullSpace(H, g, A, b)
print("x3 = \n", x3)
print("lam3 = \n", lam3)

### 7) Solve KKT system using Range-Space procedure
x4, lam4 = Solvers.EqualityQPSolverRangeSpace(H, g, A, b)
print("x4 = \n", x4)
print("lam4 = \n", lam4)

### xx) Sparse versions
x5, lam5 = Solvers.solveEqualityQP(H, g, A, b, type='LUSparse')
print("x5 = \n", x5)
print("lam5 = \n", lam5)

x6, lam6 = Solvers.solveEqualityQP(H, g, A, b, type='LDLSparse')
print("x6 = \n", x6)
print("lam6 = \n", lam6)

# Check for errors
print("x_error = \n", 
    #   np.linalg.norm(x1 + x2 + x3 + x4 + x5 - 5*x1))
      np.linalg.norm(x1 + x2 + x3 + x4 + x5 + x6 - 6*x1))
print("lam_error = \n", 
      np.linalg.norm(lam1 + lam2 + lam3 + lam4 + lam5 + lam6 - 6*lam1))

### 9+13) Performance comparison
# Problem parameters

types = ['LU', 'LDL',  'LUSparse', 'LDLSparse', 'NullSpace',
         'RangeSpace']
# types = types[-2:]
NArray = np.arange(500, 1000, 100)
times = np.zeros((len(NArray), len(types)))
outPath = f"{workDir}/timings_solvers.csv"

if 1:
    for i, type in enumerate(types):
        for j, N in enumerate(NArray):
            n = N + 1         # No. of design variables
            m = N             # No. of constraints
            H, g, A, b = setupProblem(N, uBar, d0)
            
            start_timer = timeit.default_timer()
            x, lam = Solvers.solveEqualityQP(H, g, A, b, type)        
            times[j, i] = timeit.default_timer() - start_timer
    
    df = pd.DataFrame(times, index=NArray, columns=types)
    df.to_csv(outPath)

define_plot_settings(16)
df = pd.read_csv(outPath)
fig, ax = plt.subplots()
for k, type in enumerate(types):
    if k % 2:
        ls = 'o--'
        fc = None
        ms = 6
    else:
        ls = 's-'
        fc = 'none'
        ms = 7
    ax.plot(NArray, df[type], ls, label=type, linewidth=3, 
             markerfacecolor=fc, markeredgewidth=2, markersize=ms)
ax.set_xlabel("Problem size: $N$")
ax.set_ylabel("Solution time [s]")
ax.set_xlim([np.min(NArray)*0.95, np.max(NArray)*1.1])
ax.set_ylim([0, np.max(df[types])*1.1])
# ax.set_ylim([0, 1])
ax.legend()
fig.tight_layout()
plt.savefig(f'{workDir}/timings_solvers.png')

### 10) Sparsity pattern of K
N = 100
H, g, A, b = setupProblem(N, uBar, d0)
K, r, m = Solvers.EqualityQPKKT(H, g, A, b)

fig, ax = plt.subplots()
plt.spy(K)
plt.savefig(f'{workDir}/Ksparsity_pattern.png', dpi=600)

