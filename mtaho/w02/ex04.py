import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import scipy
from scipy.optimize import minimize, Bounds
from scipy.optimize import LinearConstraint, NonlinearConstraint

projectDir = "/home/mtaho/Code/Courses/ConstrainedOptimization"
sys.path.append(os.path.realpath(f"{projectDir}"))

from mtaho.plot_settings import define_plot_settings
from mtaho.src.FiniteDifference import FD
from mtaho.src.BasicOpt import Opt

workDir = f"{projectDir}/mtaho/w02"

def contourFunHimmelblau(X, Y):
    tmp1 = X**2 + Y - 11
    tmp2 = X + Y**2 - 7
    f = tmp1**2 + tmp2**2
    return f

x0 = np.array([-3, 0])

def objFunHimmelblau(x):
    tmp1 = x[0]**2 + x[1] - 11
    tmp2 = x[0] + x[1]**2 - 7
    f = tmp1**2 + tmp2**2
    return f


# 6: Local minimizers
N = 50
x0_x = np.linspace(-5, 5, N)
x0_y = np.linspace(-5, 5, N)

def lagrangianHimmelblau_1(x, lam):
    dfdx = np.array([[4*(x[0]**2 + x[1] - 11)*x[0] + 2*(x[0] + x[1]**2 - 7)],
                     [2*(x[0]**2 + x[1] - 11) + 4*(x[0] + x[1]**2 - 7)*x[1]]])
    dcdx = np.array([[2*x[0]+4  , -4],
                     [-1        , 10]])
    dLdx = dfdx - dcdx @ lam
    return dLdx.flatten()

xsol = []
xs1Array = np.zeros((2, N*N))

def find_unique_solutions(fun, vars, lam):
    k = 0
    for x in x0_x:
        for y in x0_y:    
            xs = scipy.optimize.fsolve(fun, vars, lam)
            xs1Array[:, k] = xs
            k += 1
    xsUnique = np.unique(np.round(xs1Array, decimals=4),axis=1)
    nsol = np.shape(xsUnique)[1]
    return xsUnique, nsol

# 6.1: all constraints inactive
lam = np.array([[0, 0]]).T
xs1Unique, nsol = find_unique_solutions(lagrangianHimmelblau_1, vars, lam)

# Select feasible solutions from contour plot:
xsol.append([xs1Unique[:, 2]])
xsol.append([xs1Unique[:, 6]])
xsol.append([xs1Unique[:, 7]])

# # 6.2: c1 active and c2 inactive
# def lagrangianHimmelblau_2(x, lam):
#     dLdx = lagrangianHimmelblau_1(x, lam)
#     c1 = np.array([(x[0] + 2)**2 - x[1]])
#     return np.concatenate((dLdx, c1))

# lam = np.array([[0, 0]]).T
# xs1Unique, nsol = find_unique_solutions(lagrangianHimmelblau_2, lam)



# Plot solutions
xlim = [-5, 5]
ylim = [-5, 5]
fig, ax = Opt.contourPlot(contourFunHimmelblau,
                          xlim=xlim, ylim=ylim, vmax=200,
                          colorScale='linear', figSize=(8, 5))
ms = 8
# ax.plot(x0[0], x0[1], 'go', label='$x^0$', markersize=ms)
# for i in range(nsol):
#     ax.plot(xs1Unique[0, i], xs1Unique[1, i], 'o', label=f'i = {i}')
for i, xs in enumerate(xsol):
    ax.plot(xs[0][0], xs[0][1], 'o', label=f'$x_{i}^\\ast$')
# ax.plot(res1.x[0], res1.x[1], 'r*', label=r'$x^\ast_1$', markersize=ms)
# ax.plot(res2.x[0], res2.x[1], 'ms', markerfacecolor='none', markersize=ms, 
#         label=r'$x^\ast_2$', markeredgewidth=2)
# ax.plot(res3.x[0], res3.x[1], 'b^', markerfacecolor='none', markersize=ms, 
#         label=r'$x^\ast_3$', markeredgewidth=2)

# Draw constraints
xc = np.linspace(xlim[0], xlim[1])
alpha = 0.75
yc1 = (xc+2)**2
yc2 = (4*xc)/10
ax.fill(xc, yc1, color='grey', alpha=alpha, label='$c_1$',
        hatch='\\') 
ax.fill_between(xc, yc2, np.repeat(-5, 50), color='grey', alpha=alpha, label='$c_2$',
                hatch='/') 

ax.set_xlim(xlim)
ax.set_ylim(xlim)

ax.legend(bbox_to_anchor=(1.5, 1))
fig.tight_layout()
fig.savefig(f'{workDir}/ex04Contour.png')