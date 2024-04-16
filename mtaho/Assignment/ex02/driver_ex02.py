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

workDir = f"{projectDir}/mtaho/Assignment/ex02"

### 3) Define problem

data = sp.io.loadmat(f'{workDir}/QP_Test.mat')

# Load data
H = data['H']
g = data['g']
C = data['C']
dl = data['dl']
du = data['du']
l = data['l']
u = data['u']


# Convert problem to match custom solver interface:
#     min_x   : 0.5*x'Hx + g'x
#     s.t.    : A'x = b
#             : C'x >= d 
n, m = np.shape(C)
Cbar = np.hstack([C, -C, np.identity(n), -np.identity(n)])
dbar = np.vstack([dl, -du, l, -u])
C = Cbar.copy()
d = dbar.copy()


# Convert problem to match CVXOPT solver interface: #TODO...
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