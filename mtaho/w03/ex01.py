import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import cvxpy as cp
import scipy
from scipy.optimize import minimize, Bounds
from scipy.optimize import LinearConstraint, NonlinearConstraint

projectDir = "/home/mtaho/Code/Courses/ConstrainedOptimization"
sys.path.append(os.path.realpath(f"{projectDir}"))

from mtaho.plot_settings import define_plot_settings
from mtaho.src.FiniteDifference import FD
from mtaho.src.BasicOpt import Opt

workDir = f"{projectDir}/mtaho/w03"

def EqualityQPSolver(H, g, A, b):
    """ Solve the KKT system for an equality constrained QP"""

    # Initialize
    n = np.shape(H)[0]
    m = np.shape(A)[1]
    K = np.zeros((n + m, n + m))
    r = np.zeros((n + m, 1))
    u = np.zeros((n + m, 1))

    # Build KKT matrix and right-hand side
    K[0:n, 0:n] = H
    K[n:n+m, 0:n] = -A.T
    K[0:n, n:n+m] = -A

    r[0:n, 0:1] = -g
    r[n:n+m, 0:1] = -b

    # Solve system using LDL factorization: # System:           Ku = r  
        # Factorization:  PKP' = LDL' (Px = x(p))
    L, D, p = scipy.linalg.ldl(K, lower=True)           
    w = np.linalg.solve(L, r)            # Solve:            Lw = Pr
    v = np.linalg.solve(D, w)               # Solve:            Dv = w
    u = np.linalg.solve(L.T, v)          # Solve:           L'u = y

    # Map solution to design variables and Lagrange multipliers
    x = u[0:n]
    lam = u[n:n+m]

    return x, lam

# # 1.4) Test solver
H = np.array([[6, 2, 1],
              [2, 5, 2],
              [1, 2, 4]], dtype=np.float64)
g = np.array([[-8, -3, -3]], dtype=np.float64).T
A = np.array([[1, 0],
              [0, 1],
              [1, 1]], dtype=np.float64)
b = np.array([[3, 0]]).T

print("H = \n", H)
print("g = \n", g)
print("A = \n", A)
print("b = \n", b)

x_s, lam_s = EqualityQPSolver(H, g, A, b)

# print("x = \n", x_s)
# print("lam = \n", lam_s)

# 1.5) Generate and solve random convex QP
n = 20
m = 5      # m <= n

np.random.seed(1)
h = np.random.randn(n, n)
H = h.T @ h

A = np.random.randn(n, m)

x = np.random.randn(n, 1)
lam = np.random.randn(m, 1)**2

g = A @ lam - H @ x
b = A.T @ x

x_s, lam_s = EqualityQPSolver(H, g, A, b)

print("x_s - x = \n", x_s - x)
print("lam_s - lam = \n", lam_s - lam)

# 1.7) Sensitivities

def EqualityQPSens(H, A):
    """ Form the sensitivity matrix of the equality constrained QP problem """

    # Initialize
    n = np.shape(H)[0]
    m = np.shape(A)[1]
    K = np.zeros((n + m, n + m))
    
    # Build KKT matrix and right-hand side
    K[0:n, 0:n] = H
    K[n:n+m, 0:n] = -A.T
    K[0:n, n:n+m] = -A

    # Compute sensitivity matrix
    dudr = - np.linalg.inv(K)
    
    return dudr

dudr = EqualityQPSens(H, A)

# 1.10) Solving the dual QP:
def EqualityQPDualSolver(H, g, A, b):
    """ Solve the euality constrained dual QP"""
    Hinv = np.linalg.inv(H)
    P = A.T @ Hinv @ A
    d = b + A.T @ Hinv @ g
    lam = np.linalg.solve(P, d)
    x = Hinv @ ( A @ lam - g )
    return x, lam


x_sd, lam_sd = EqualityQPDualSolver(H, g, A, b)

print("x_sd - x = \n", x_sd - x)
print("lam_sd - lam = \n", lam_sd - lam)

print("hello")