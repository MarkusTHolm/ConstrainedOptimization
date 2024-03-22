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
np.set_printoptions(threshold=sys.maxsize)

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
if 1: # Generate random problem that is well-posed
    
    # Settings
    n = 4
    m = 2
    seed = 10
    np.random.seed(seed)

    # Define random problem
    A = np.random.randn(m, n)
    
    x = np.zeros((n, 1))
    x[0:m, 0:1] = np.abs(np.random.randn(m, 1))
    
    s = np.zeros((n, 1))
    s[m:n, 0:1] = np.abs(np.random.randn(n-m, 1))

    lam = np.random.randn(m, 1)

    g = A.T @ lam + s
    b = A@x

    if 1:
        # Solve problem using Slack variables
        # min_{x,t} : [0]'[x ]
        #             [1] [t ]
        #             [0] [s1]
        #             [0] [s2]
        #       s.t.  :             [x ] 
        #               [ A e  -I 0][t ]   [b]
        #               [-A e  0 -I][s1] = [-b]
        #                           [s2] 
        n1 = (n+1+2*m)
        m1 = 2*m
        e = np.ones((m, 1))

        g1 = np.zeros((n1, 1))
        g1[n] = 1

        A1 = np.zeros((m1, n1))
        A1[0:m,  0:n] = A
        A1[m:m1, 0:n] = -A
        A1[0:m,  n:n+1] = e
        A1[m:m1, n:n+1] = e
        A1[0:m:, n+1:n+1+m] = -np.identity(m)
        A1[m:m1:,n+1+m:n1] = -np.identity(m)

        b1 = np.zeros((m1, 1))
        b1[0:m] = b
        b1[m:m1] = -b

        x01 = np.zeros((n1, 1))

        t = np.max(np.abs(b))
        s1 = t*e - b
        s2 = t*e + b
        x1 = np.zeros((n1, 1))
        x1[n] = t
        x1[n+1:n+1+m] = s1
        x1[n+1+m:n1] = s2   
        
    sol = Solvers.LPSolverRevisedSimplex(g1, A1, b1, x1)
    x0 = sol['x'][0:n]
    print(f"err_inf = {np.linalg.norm(x0-x,np.inf)}")

else: 
    # Problem from Wikipedia: https://en.wikipedia.org/wiki/Revised_simplex_method
    # g = np.array([[-2, -3, -4, 0, 0]], dtype=np.float64).T
    # A = np.array([[3, 2, 1, 1, 0],
    #               [2, 5, 3, 0, 1]], dtype=np.float64)
    # b = np.array([[10, 15]], dtype=np.float64).T
    # x0 = np.array([[0, 0, 0, 10, 15]], dtype=np.float64).T

    # Problem from Nocedal and Wright
    g = np.array([[-3, -2, 0, 0]], dtype=np.float64).T
    A = np.array([[1, 1,   1, 0],
                  [2, 0.5, 0, 1]], dtype=np.float64)
    b = np.array([[5, 8]], dtype=np.float64).T
    x0 = np.array([[0, 0, 5, 8]], dtype=np.float64).T

## Test method

sol = Solvers.LPSolverRevisedSimplex(g, A, b, x0)
xs = sol['x']

print(f"xs = \n{xs}")
