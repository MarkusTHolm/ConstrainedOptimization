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

workDir = f"{projectDir}/mtaho/w08"

def randomQP(n, alpha, density, seed):
    # Generates data for a random convex Qp
    #
    # Problem: 
    # min_x : 0.5*x'Hx + g'x
    # s.t.  : bl <= A'x <= bu
    #       : l  <= u   <= x
    #
    # Inputs:
    #   n       : Number of variables
    #   alpha   : Regularization factor. alpha > 0
    #   density : Density of sparse matrix. 0 < density < 1
    np.random.seed(seed=seed)    
    rng = np.random.default_rng(seed=seed)
    rvs = sp.stats.norm(loc=1, scale=1).rvs

    m = int(n*10)
    A = sp.sparse.random(n, m, density, random_state=rng,
                         data_rvs=rvs)
    bl = -np.random.rand(m, 1)
    bu = np.random.rand(m, 1)
    M = sp.sparse.random(n, n, density, random_state=rng,
                         data_rvs=rvs)
    H = M@M.T + alpha*sp.sparse.eye(n, n)
    g = np.random.randn(n, 1)
    l = -np.ones((n, 1))
    u = np.ones((n, 1))

    return H, g, A, bl, bu, l, u

# Problem: 
# min_x : 0.5*x'Hx + g'x
# s.t.  : bl <= A'x <= bu
#       : l  <= u   <= x
n = 30
alpha = 0.2
density = 1
seed = 100
H, g, A, bl, bu, l, u = randomQP(n, alpha, density, seed)
x0 = np.zeros((n, 1))

### Interior point set algorithm for convex QPs:

## Interface:
# min_x : 0.5*x'Hx + g'x
# s.t.  : A'x = b
#       : C'x >= d

# Convert problem to match solver:
Abar = np.hstack([A.toarray(), -A.toarray(), np.identity(n), -np.identity(n)])
bbar = np.vstack([bl, -bu, l, -u])
C = Abar.copy()
d = bbar.copy()

sol = Solvers.QPSolverInteriorPoint(H, g, C, d, x0=x0)
xint = sol['x']
# print("x = \n", xint)

### Active set algorithm for convex QPs:

## Interface:
# min_x : 0.5*x'Hx + g'x
# s.t.  : A'x + b >= 0
A = Abar.copy()
b = -bbar.copy()

# Save data
mdic = {"H": H, "g": g, "A": A, "b": b}
dir = '/home/mtaho/OneDrive/PhD/08_Courses/ConstrainedOptimization/Litterature/w08'
sp.io.savemat(f"{dir}/testPy.mat", mdic)

sol = Solvers.QPSolverInequalityActiveSet(H, g, A, b, x0)
xact = sol['x']

err = np.linalg.norm(xint-xact, np.inf)
print(f"Check matching solution: |xint-xact|_inf = {err:1.3e}")
