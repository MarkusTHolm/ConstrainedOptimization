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

workDir = f"{projectDir}/mtaho/w04"

def contourFunHimmelblau(X, Y):
    tmp1 = X**2 + Y - 11
    tmp2 = X + Y**2 - 7
    f = tmp1**2 + tmp2**2
    return f

x0 = np.array([-4, -1])
xLower = [-5, -5]
xUpper = [5, 5]
bounds = Bounds(xLower, xUpper)

####################################################################################
# a) With gradients of objective function and constraints
####################################################################################

## Objective function and gradients
def objFunGradHimmelblau(x):
    tmp1 = x[0]**2 + x[1] - 11
    tmp2 = x[0] + x[1]**2 - 7
    f = tmp1**2 + tmp2**2
    df = np.array([4*(x[0]**2 + x[1] - 11)*x[0] + 2*(x[0] + x[1]**2 - 7),
                   2*(x[0]**2 + x[1] - 11) + 4*(x[0] + x[1]**2 - 7)*x[1]])
    return (f, df)

## Constraints

# c1 = (x1 + 2)^2 - x2 >= 0 (Nonlinear constraint)
def cons_f(x):
    return (x[0] + 2)**2 - x[1]
def cons_J(x):
    return [2*(x[0] + 2), - 1]

nonlinear_constraint = NonlinearConstraint(cons_f, 0, np.inf, jac=cons_J)

# c2 = -4*x1 + 10*x2 >= 0 (Linear constraint)
A = np.array([[-4, 10]])
linear_constraint = LinearConstraint(A, [0], [np.inf])

print("Results version a): Only gradients ")
options = {}
tols = np.logspace(-16, -3, 10)
for tol in tols:
    options["tol"] = float(tol)
    res_a = cyipopt.minimize_ipopt(objFunGradHimmelblau, x0, jac=True,
                                constraints=[linear_constraint, nonlinear_constraint],
                                options=options)
    print(f"xs = {res_a.x}, rel_tol = {tol:1.3e}, nit = {res_a.nit}")

####################################################################################
# b) With gradients and Hessian of objective function and constraints
####################################################################################

## Objective function and gradients
def fun(x):
    tmp1 = x[0]**2 + x[1] - 11
    tmp2 = x[0] + x[1]**2 - 7
    f = tmp1**2 + tmp2**2
    return f

def fun_jac(x):
    df = np.array([4*(x[0]**2 + x[1] - 11)*x[0] + 2*(x[0] + x[1]**2 - 7),
                   2*(x[0]**2 + x[1] - 11) + 4*(x[0] + x[1]**2 - 7)*x[1]])
    return df

def fun_hess(x):
    d2f = np.array([[12*x[0]**2 + 4*x[1] - 42 ,4*x[0] + 4*x[1]         ],
                    [4*x[0] + 4*x[1]          ,12*x[1]**2 + 4*x[0] - 26]])
    return d2f

## Constraints

# c1 = (x1 + 2)^2 - x2 >= 0 (Nonlinear constraint)
def cons1_f(x):
    return (x[0] + 2)**2 - x[1]
def cons1_J(x):
    return [2*(x[0] + 2), - 1]
def cons1_H(x, a):
    return a*np.array([[2, 0], [0, 0]])

# c2 = -4*x1 + 10*x2 >= 0 (Linear constraint)
def cons2_f(x):
    return A@x
def cons2_J(x):
    return [-4, 10]
def cons2_H(x, a):
    return a*np.zeros((2, 2))

c1 = {'type': 'ineq', 'fun': cons1_f, 'jac': cons1_J, 'hess': cons1_H}
c2 = {'type': 'ineq', 'fun': cons2_f, 'jac': cons2_J, 'hess': cons2_H}

print("Results of version b): Hessian ")
options = {}
options['disp'] = 5
tols = np.logspace(-16, -3, 10)
for tol in tols:
    options["tol"] = float(tol)
    res_b = cyipopt.minimize_ipopt(fun, x0=x0, jac=fun_jac, hess=fun_hess,
                                   constraints=[c1, c2],
                                   options=options)

    print(f"xs = {res_b.x}, rel_tol = {tol:1.3e}, nit = {res_b.nit}")

####################################################################################
# Plot solutions
####################################################################################

xlim = [-5, 5]
ylim = [-5, 5]
fig, ax = Opt.contourPlot(contourFunHimmelblau,
                          xlim=xlim, ylim=ylim, vmax=200,
                          colorScale='linear', figSize=(8, 5))
ms = 8

# Plot solutions
ax.plot(x0[0], x0[1], 'ko', label='$x^0$', markersize=ms)
ax.plot(res_a.x[0], res_a.x[1], 'b^', markerfacecolor='none', markersize=ms, 
        label=r'$x^\ast_a$', markeredgewidth=2)
ax.plot(res_b.x[0], res_b.x[1], 'rs', markerfacecolor='none', markersize=ms, 
        label=r'$x^\ast_b$', markeredgewidth=2)
# Draw constraints
xc = np.linspace(xlim[0], xlim[1])
alpha = 0.5
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
fig.savefig(f'{workDir}/ex01.png')