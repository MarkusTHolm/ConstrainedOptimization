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


P_max = np.vstack((Pd_max, Pg_max))

# Convert to the form:
#     min_x   : g'x
#     s.t.    : A'x <= b
#             : l <= x <= u

nd = np.shape(U)[0]
ng = np.shape(C)[0]
n = ng+nd
m = 1
ed = np.ones((nd, 1))
eg = np.ones((ng, 1))

# x = [P_d]
#     [P_g]
g = np.vstack((-U, C))
A = np.vstack((ed, -eg))
b = 0
l = np.zeros((n, 1))
u = np.vstack((Pd_max, Pg_max))

x0 = np.ones((n, 1))

#########################################################################################
# Library solution 
#########################################################################################
# Convert problem to match SciPy solver interface:
#     min_x   : c'x
#     s.t.    : A_ub x <= b_ub
#             : A_eq x <= b_eq
#             : l <= x <= u
c = g
A_eq = A.T
b_eq = b
bounds = np.hstack((l, u))

N = 10
times = np.zeros((N))
for i in range(N):
    start_timer = timeit.default_timer()
    res = sp.optimize.linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds,
                            method='highs-ipm', 
                            options={"disp":False,
                                    "presolve":False,
                                    "primal_feasibility_tolerance":1e-10,
                                    "dual_feasibility_tolerance":1e-10,
                                    "ipm_optimality_tolerance":1e-12})
    times[i] = timeit.default_timer() - start_timer
print(f"Average solution time: {np.average(times):1.4f}, "
      f"iter = {res['nit']}")


x = res['x']
P = x
Pd = P[0:nd]
Pg = P[nd:n]
mu = -res['eqlin']['marginals'][np.newaxis].T
lam = -res['upper']['marginals'][np.newaxis].T

# Print solution statistics
L =  g.T@x - mu.T@(A.T@x - b)  - lam.T@x
dLdx = g - A@mu - lam
dLdmu = -(A.T@x - b)
print(f"dLdx/L = {np.linalg.norm(dLdx, np.inf)/np.abs(L[0]):1.3e}, "
      f"dLdz/d = {np.max(dLdmu):1.3e}")

#########################################################################################
# Custom revised simplex method
#########################################################################################
# Convert problem to standard form to match custom solver interface:
# min_x : g'x
# s.t.  : Ax = b
#       :  x >= 0
I = np.identity(n)

nS = 4*n
mS = m+2*n
gBar = np.zeros((nS, 1))
gBar[0:n] = -g
gBar[n:2*n] = g

ABar = np.zeros((mS, 4*n))
ABar[0:m        ,0:n] = -A.T
ABar[m:m+n      ,0:n] = -I
ABar[m+n:m+2*n  ,0:n] = -I

ABar[0:m        ,n:2*n] = A.T
ABar[m:m+n      ,n:2*n] = I
ABar[m+n:m+2*n  ,n:2*n] = I

ABar[m:m+n      ,2*n:3*n] = -I

ABar[m+n:m+2*n  ,3*n:4*n] = I

bBar = np.zeros((mS, 1))
bBar[0:m] = b
bBar[m:m+n] = l
bBar[m+n:m+2*n] = u

x0 = np.ones((nS, 1))

# Find initial feasible point by solving:
# min_{x,t} : [0]'[x ]
#             [1] [t ]
#             [0] [s1]
#             [0] [s2]
#       s.t.  :             [x ] 
#               [ A e  -I 0][t ]   [b]
#               [-A e  0 -I][s1] = [-b]
#                           [s2] 
n1 = (nS+1+2*mS)
m1 = 2*mS
e = np.ones((mS, 1))
I = np.identity(mS)

g1 = np.zeros((n1, 1))
g1[nS] = 1

A1 = np.zeros((m1, n1))
A1[0:mS,  0:nS] = ABar
A1[mS:m1, 0:nS] = -ABar
A1[0:mS,  nS:nS+1] = e
A1[mS:m1, nS:nS+1] = e
A1[0:mS:, nS+1:nS+1+mS] = -I
A1[mS:m1:,nS+1+mS:n1] = -I

b1 = np.zeros((m1, 1))
b1[0:mS] = bBar
b1[mS:m1] = -bBar

t = np.max(np.abs(bBar))
s1 = t*e - bBar
s2 = t*e + bBar
x0P1 = np.zeros((n1, 1))
x0P1[nS] = t
x0P1[nS+1:nS+1+mS] = s1
x0P1[nS+1+mS:n1] = s2   

res01 = sp.optimize.linprog(g1, A_eq=A1, b_eq=b1, x0=x0P1,
                            method='revised simplex', 
                            options={"disp":False})

sol = Solvers.LPSolverRevisedSimplex(g1, A1, b1, x0P1)

x0 = res01['x'][0:nS][np.newaxis].T
x0[np.isclose(x0, 0)] = 0

# res02 = sp.optimize.linprog(gBar, A_eq=ABar, b_eq=bBar, x0=x0,
#                             method='revised simplex', 
#                             options={"disp":False})

# sol = Solvers.LPSolverRevisedSimplex(gBar, ABar, bBar, x0)


# xs = sol['x']
# print(f"xs = \n{xs}")


low = np.zeros((4*n, 1))
upp = 1e6*np.ones((4*n, 1))
bounds = np.hstack((low, upp))
res2 = sp.optimize.linprog(gBar, A_eq=ABar, b_eq=bBar, bounds=bounds,
                        method='highs-ipm', 
                        options={"disp":False,
                                "presolve":False,
                                "primal_feasibility_tolerance":1e-10,
                                "dual_feasibility_tolerance":1e-10,
                                "ipm_optimality_tolerance":1e-12})
x2 = res2['x'][n:2*n]
print(f"Solution error: {np.linalg.norm(x2-x,np.inf)}")

#########################################################################################
# Plot solution 
#########################################################################################

# Plot solution (bar chart)
define_plot_settings(18)
fig, axs = plt.subplots(1, 2, figsize=(12,4), gridspec_kw={'width_ratios': [1, 2]})
idxPd = np.arange(len(Pd))+1
idxPg = np.arange(len(Pg))+1

axs[0].bar(idxPg,Pg, color='tab:orange')
axs[0].plot(idxPg,Pg_max.flatten(), '_', color='k', markersize=12)

axs[1].bar(idxPd,Pd)
axs[1].plot(idxPd,Pd_max.flatten(), '_', color='k', markersize=12)

axs[0].set_xticks(idxPg,idxPg.astype(str),rotation=60)
axs[1].set_xticks(idxPd,idxPd.astype(str),rotation=60)

axs[0].set_xlabel('Generators: $g$')
axs[1].set_xlabel('Demands: $d$')

axs[0].set_ylabel("Power generated: $p_g$ [MW]")
axs[1].set_ylabel("Power used: $p_d$ [MW]")

axs[0].set_xlim([0, np.max(idxPg)+1])
axs[1].set_xlim([0, np.max(idxPd)+1])

fig.tight_layout()
fig.savefig(f"{workDir}/solution.png")

#########################################################################################
# Plot supply-demand curve
#########################################################################################

supply = np.zeros((ng, 2))   # supply(quantity, price)
demand = np.zeros((nd, 2))   # demand(quantity, price)

# Sort according to bid prices and offer prices

# Sort bid prices from highest to lowest (sell to highest bidder first)
sort_idx_U = np.argsort(U.flatten())[::-1]
U_sorted =  U[sort_idx_U]
Pd_max_sorted = Pd_max[sort_idx_U]

# Sort offer prices from lowest to highest (offer lowest price first)
sort_idx_C = np.argsort(C.flatten())
C_sorted =  C[sort_idx_C]
Pg_max_sorted = Pg_max[sort_idx_C]

# Determine demand quantity and price
quantity = 0
price = 0
for i in range(nd):
    pdMax = Pd_max_sorted[i]
    quantity += pdMax
    price = U_sorted[i]
    demand[i, 0] = quantity[0]
    demand[i, 1] = price[0]

# Determine supply quantity and price
quantity = 0
price = 0
for i in range(ng):
    pgMax = Pg_max_sorted[i]
    quantity += pgMax
    price = C_sorted[i]
    supply[i, 0] = quantity[0]
    supply[i, 1] = price[0]

# Solution
PdSol = P[0:nd]
PgSol = P[nd:n]
demandSol = np.array([np.sum(PdSol), (U.T@PdSol)[0]])
supplySol = np.array([np.sum(PgSol), (C.T@PgSol)[0]])

define_plot_settings(18)
fig, ax = plt.subplots(figsize=(9, 4))

# Plot supply-demand curves
ax.step(supply[:, 0], supply[:, 1], label='Supply')
ax.step(demand[:, 0], demand[:, 1], label='Demand')

ax.hlines(mu, 0, 10e3, color='black', linestyle='--',
          label='Market clearing price')

ax.set_xlabel(r"Energy quantity [MW]")
ax.set_ylabel(r"Price [\euro/MW]")
ax.legend(bbox_to_anchor=(1, 1), loc="upper left")
ax.set_xlim([0, quantity])
fig.tight_layout()

fig.savefig(f"{workDir}/supply_demand.png", dpi=600)
# plt.show(block=False)

print(res)






