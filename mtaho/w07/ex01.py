import matplotlib.pyplot as plt
from cycler import cycler
import numpy as np
import os
import sys
import cvxpy as cp
import cvxopt 
import scipy
from scipy.optimize import minimize, Bounds
from scipy.optimize import LinearConstraint, NonlinearConstraint

projectDir = "/home/mtaho/Code/Courses/ConstrainedOptimization"
# projectDir = "C:/Users/marku/Programming/ConstrainedOptimization"
sys.path.append(os.path.realpath(f"{projectDir}"))

from mtaho.plot_settings import define_plot_settings
from mtaho.src.FiniteDifference import FD
from mtaho.src.BasicOpt import Opt
from mtaho.src.Solvers import Solvers

workDir = f"{projectDir}/mtaho/w06"

## Problem
# min_x : c'x
# s.t.  : Ax = b
#       : x >= 0

## Revised simplex method for LP in standard form:

# Interface:
# min_x : c'x
# s.t.  : Ax = b
#       : x >= 0

# 
if 0: # Generate random problem that is well-posed
    n = 6
    m = 3
    seed = 10

    np.random.seed(seed)

    x = np.zeros((n, 1))
    x[0:m, 0:1] = np.abs(np.random.randn(m, 1))

    s = np.zeros((n, 1))
    s[m:n, 0:1] = np.abs(np.random.randn(n-m, 1))

    lam = np.random.randn(m, 1)
    A = np.abs(np.random.randn(m, n))

    c = A.T@lam + s
    b = A@x
else: # Problem from Nocedal and Wright
    c = np.array([[-3, -2, 0, 0]], dtype=np.float64).T
    A = np.array([[1, 1,   1, 0],
                  [2, 0.5, 0, 1]], dtype=np.float64)
    b = np.array([[5, 8]], dtype=np.float64).T

## Test method
x0 = np.array([[0, 0, 5, 8]], dtype=np.float64).T
sol = Solvers.LPSolverRevisedSimplex(c, A, b, x0)

