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

workDir = f"{projectDir}/mtaho/Assignment/ex03"

### 3) Define problem

data = sp.io.loadmat(f'{workDir}/LP_Test.mat')

# Load data
C = data['C'].astype(np.float64)
Pd_max = data['Pd_max'].astype(np.float64)
Pg_max = data['Pg_max'].astype(np.float64)
U = data['U'].astype(np.float64)

# Convert problem to match custom solver interface:
#     min_x   : 0.5*x'Hx + g'x
#     s.t.    : A'x = b
#             : C'x >= d 
n, mI = np.shape(C)
CBar = np.hstack([C, -C, np.identity(n), -np.identity(n)])
dBar = np.vstack([dl, -du, l, -u])
n, m = np.shape(CBar)

x0 = np.ones((n, 1))

def printStats(x, z):
    L = 0.5*x.T@H@x + g.T@x - z.T@(CBar.T@x - dBar)
    dLdx = H@x + g - CBar@z
    dLdz = -(CBar.T@x - dBar)
    relative_gap = np.linalg.norm((CBar.T@x - dBar)*z/L, np.inf)
    print(f"dLdx/L = {np.linalg.norm(dLdx, np.inf)/np.abs(L[0, 0]):1.3e}, "
          f"dLdz/d = {np.max(dLdz)/np.linalg.norm(dBar, np.inf):1.3e}, "
          f"relative_gap = {relative_gap:1.3e}")
    return 

# Interior point solver
print("\n Interior point solver:")
N = 1
times = np.zeros((N))
for i in range(N):
    start_timer = timeit.default_timer()
    solIP = Solvers.QPSolverInteriorPoint(H, g, CBar, dBar)
    times[i] = timeit.default_timer() - start_timer
print(f"Average solution time: {np.average(times):1.4f}, "
      f"iter = {solIP['iter']}")
xIP = solIP['x']
zIP = solIP['z']
printStats(xIP, zIP)

# Active set method

# Find feasible initial point
nx = n
nz = m
nLP = nx + nz
mLP = m + nz

# Feasible initial point for Phase 1 LP
xTilde = x0
gamma = 1
x0LP = cvxopt.matrix(0.0, (nLP, 1))
z0 = np.maximum(dBar - CBar.T@xTilde, 0)
x0LP[0:nx] = cvxopt.matrix(xTilde)
x0LP[nx:nLP] = cvxopt.matrix(z0)

# LP solver interface:
# minimize    c'*x
# subject to  G*x <= h
c = cvxopt.matrix(0.0, (nLP, 1))
c[nx:nLP] = 1.0

G = cvxopt.matrix(0.0, (mLP, nLP))
G[0:nz, 0:nx] = cvxopt.matrix(CBar.T)
Gzx = G[0:nz, nx:]
Gzx[::nz+1] = gamma
G[:nz, nx:] = Gzx
Gzz = G[nz:, nx:]
Gzz[::nz+1] = 1.0
G[nz:, nx:] = Gzz

h = cvxopt.matrix(0.0, (mLP, 1))
h[0:nz] = cvxopt.matrix(dBar)

sol = cvxopt.solvers.lp(c, G, h, options={'show_progress': False})

# x0 = sol['x'][0:nx][np.newaxis].T
x0 = np.array(sol['x'])[0:nx]
# print(x0)

print("\n Active set solver:") 
N = 1
times = np.zeros((N))
for i in range(N):
    start_timer = timeit.default_timer()
    solAct = Solvers.QPSolverInequalityActiveSet(H, g, CBar, dBar, x0=x0)
    times[i] = timeit.default_timer() - start_timer
print(f"Average solution time: {np.average(times):1.4f}, "
      f"iter = {solAct['iter']}")
xAct = solAct['x']
zAct = solAct['z']
printStats(xAct, zAct)

# Convert problem to match CVXOPT solver interface: 
# minimize    (1/2)*x'*P*x + q'*x
# subject to  G*x <= h
#             A*x = b.
P = cvxopt.matrix(H)
q = cvxopt.matrix(g)
G = cvxopt.matrix(-CBar.T)
h = cvxopt.matrix(-dBar)

print("\n CVXopt solver")
N = 1
times = np.zeros((N))
for i in range(N):
    start_timer = timeit.default_timer()
    solCVX = cvxopt.solvers.qp(P, q, G, h, 
                               options={'show_progress': False})
    times[i] = timeit.default_timer() - start_timer
print(f"Average solution time: {np.average(times):1.4f}, "
      f"iter = {solCVX['iterations']}")
xCVX = np.array(solCVX['x'])
zCVX = np.array(solCVX['z'])
printStats(xCVX, zCVX)
outData = {"U": xCVX}
sp.io.savemat(f'{workDir}/solCVX.mat', outData)

# Compare solutions

print("\n Relative errors")
errIPx = np.linalg.norm((xIP-xCVX), np.inf)/np.linalg.norm(xCVX, np.inf)
errIPz = np.linalg.norm((zIP-zCVX), np.inf)/np.linalg.norm(zCVX, np.inf)
errActx = np.linalg.norm(xAct-xCVX, np.inf)/np.linalg.norm(xCVX, np.inf)
errActz = np.linalg.norm(zAct-zCVX, np.inf)/np.linalg.norm(zCVX, np.inf)
print(f"Interior point: eps_x = {errIPx:1.4e}, eps_z = {errIPz:1.4e}")
print(f"Active set: eps_x = {errActx:1.4e}, eps_z = {errActz:1.4e}")

print("")

# Plot KKT residuals of the IP method
residuals = solIP['residuals']

define_plot_settings(22)
fig, ax = plt.subplots(figsize=(8,5))

kMax = solIP['iter']+1
iter = np.arange(1, kMax)
lw = 3
ax.semilogy(iter, residuals[0, 1:kMax].T, 'o-',
            label=r'$\lVert r_L \rVert_{\infty}$', linewidth=lw)
ax.semilogy(iter, residuals[2, 1:kMax].T, '^--',
             label=r'$\lVert r_C \rVert_{\infty}$', linewidth=lw)
ax.semilogy(iter, residuals[3, 1:kMax].T, 's-',
            label=r'$\lVert r_{sz} \rVert_{\infty}$', linewidth=lw)

ax.grid()
ax.set_xlabel("Iteration: $k$")
ax.set_ylabel("Residuals")
ax.set_xticks(iter)
# ax.set_ylim([0, 1])
ax.legend()
fig.tight_layout()
plt.savefig(f'{workDir}/KKT_residuals.eps')


# print(xCVX)