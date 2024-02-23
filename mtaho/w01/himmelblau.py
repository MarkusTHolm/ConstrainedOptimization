import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from scipy.optimize import minimize, Bounds
from scipy.optimize import LinearConstraint, NonlinearConstraint

projectDir = "/home/mtaho/Code/Courses/ConstrainedOptimization"
sys.path.append(os.path.realpath(f"{projectDir}"))

from mtaho.plot_settings import define_plot_settings
from mtaho.src.FiniteDifference import FD
from mtaho.src.BasicOpt import Opt

workDir = f"{projectDir}/mtaho/w01"

def contourFunHimmelblau(X, Y):
    tmp1 = X**2 + Y - 11
    tmp2 = X + Y**2 - 7
    f = tmp1**2 + tmp2**2
    return f

x0 = np.array([-3, 0])
xLower = [-5, -5]
xUpper = [5, 5]
bounds = Bounds(xLower, xUpper)

# Version 1 - No gradients and no constraints
def objFunHimmelblau(x):
    tmp1 = x[0]**2 + x[1] - 11
    tmp2 = x[0] + x[1]**2 - 7
    f = tmp1**2 + tmp2**2
    return f

print("\n Results of version 1: ")
res1 = minimize(objFunHimmelblau, x0,
               method='trust-constr', 
               options={'verbose': 1}, bounds=bounds)

# Version 2 - With gradients
def objFunGradHimmelblau(x):
    tmp1 = x[0]**2 + x[1] - 11
    tmp2 = x[0] + x[1]**2 - 7
    f = tmp1**2 + tmp2**2
    df = np.array([4*(x[0]**2 + x[1] - 11)*x[0] + 2*(x[0] + x[1]**2 - 7),
                   2*(x[0]**2 + x[1] - 11) + 4*(x[0] + x[1]**2 - 7)*x[1]])
    return (f, df)

print("\n Results of version 2: ")
res2 = minimize(objFunGradHimmelblau, x0,jac=True,
               method='trust-constr', 
               options={'verbose': 1}, bounds=bounds)

# Version 3 - With gradients and constraints

# c1 = (x1 + 2)^2 - x2 >= 0 (Nonlinear constraint)
def cons_f(x):
    return (x[0] + 2)**2 - x[1]
def cons_J(x):
    return [2*(x[0] + 2), - 1]
nonlinear_constraint = NonlinearConstraint(cons_f, 0, np.inf, jac=cons_J)

# c2 = -4*x1 + 10*x2 >= 0 (Linear constraint)
A = [-4, 10]
linear_constraint = LinearConstraint(A, [0], [np.inf])

print("\n Results of version 3: ")
res3 = minimize(objFunGradHimmelblau, x0, jac=True,
                method='trust-constr',
                constraints=[linear_constraint, nonlinear_constraint], 
                options={'verbose': 1}, bounds=bounds)


# Plot solutions
xlim = [-5, 5]
ylim = [-5, 5]
fig, ax = Opt.contourPlot(contourFunHimmelblau,
                          xlim=xlim, ylim=ylim, vmax=200,
                          colorScale='linear', figSize=(8, 5))
ms = 8
ax.plot(x0[0], x0[1], 'go', label='$x^0$', markersize=ms)
ax.plot(res1.x[0], res1.x[1], 'r*', label=r'$x^\ast_1$', markersize=ms)
ax.plot(res2.x[0], res2.x[1], 'ms', markerfacecolor='none', markersize=ms, 
        label=r'$x^\ast_2$', markeredgewidth=2)
ax.plot(res3.x[0], res3.x[1], 'b^', markerfacecolor='none', markersize=ms, 
        label=r'$x^\ast_3$', markeredgewidth=2)

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
fig.savefig(f'{workDir}/himmelblauContour.png')