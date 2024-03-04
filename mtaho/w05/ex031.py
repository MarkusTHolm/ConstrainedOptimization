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


## Markowitz Portfolio Optimization

# 1) Given return R, we can find the portfolio with minimum risk by solving

# min_x :      f(x) = 0.5*x'Hx
# s.t.  :      mu'x = R
#       : np.sum(x) = 1
#       :        x >= 0       


H = np.array([[2.30, 0.93, 0.62, 0.74, -0.23],
              [0.93, 1.40, 0.22, 0.56, 0.26],
              [0.62, 0.22, 1.80, 0.78, -0.27],
              [0.74, 0.56, 0.78, 3.40, -0.56],
              [-0.23, 0.26, -0.27, -0.56, 2.60]])

mu = np.array([15.10, 12.50, 14.70, 9.02, 17.68])

# 2) The minimal return is found by investing everything i no. 4 (x4=1)
#    and the maximal return is found by investing everything in no. 5 (x5=1)

# 3) Use of QP solver to find a portfolio with return R=10.0 and minimal risk:

R = 10.0

P = cvxopt.matrix(H)
q = cvxopt.matrix([0., 0., 0., 0., 0.])

n = 5
G = cvxopt.matrix(0.0, (n,n))
G[::n+1] = -1.0

h = cvxopt.matrix(0.0, (n,1))

A = cvxopt.matrix(0.0, (2, n))
A[0, 0:n] = cvxopt.matrix([[1.0, 1.0, 1.0, 1.0, 1.0]]).T
A[1, 0:n] = cvxopt.matrix(mu).T

b = cvxopt.matrix(0.0, (2, 1))
b[0] = cvxopt.matrix(1.0)
b[1] = cvxopt.matrix(R)


portfolio = cvxopt.solvers.qp(P, q, G, h, A, b)


print(f"Optimal portfolio for R=10.0 is: x = \n{portfolio['x']}")

# 4) Efficient frontier and optimal portfolio as a function of return

N = 100
returns = np.linspace(np.min(mu), np.max(mu), N)

risks = np.zeros((N))
portfolios = np.zeros((n, N))

for i, R in enumerate(returns):
    b[1] = cvxopt.matrix(R)
    portfolio = cvxopt.solvers.qp(P, q, G, h, A, b)
    xs = portfolio['s']
    risks[i] = xs.T @ H @ xs
    portfolios[:, i:i+1] = xs

# Plot efficient frontier
define_plot_settings(16)
fig, ax = plt.subplots()
ax.plot(returns, risks)
ax.plot(mu, H.diagonal(), 'o')
for i, asset in enumerate([f"Asset {i+1}" for i in range(n)]):
        ax.text(mu[i]+0.3, H.diagonal()[i], asset)
ax.set_xlabel('Return: $R$')
ax.set_ylabel('Risk: $V(R)$')
fig.tight_layout()
plt.savefig(f'{workDir}/efficientFrontier31.png')

# Plot portfolios
define_plot_settings(16)
rows = 3
cols = 2
fig, axs = plt.subplots(rows, cols)
k = 0
for i in range(rows):
    for j in range(cols):
        
        if k >= n:
             break

        axs[i, j].plot(returns, portfolios[k, :])

        # if i == rows-1:
        axs[i, j].set_xlabel('Return')
        axs[i, j].set_ylabel(f"Asset {k+1}")
        
        k += 1 

fig.delaxes(axs[2, 1])

plt.tight_layout()
plt.savefig(f'{workDir}/portfolios31.png')
