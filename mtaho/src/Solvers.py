import numpy as np
import scipy
import cvxopt
import cvxopt.cholmod
# import sksparse
# import sksparse.cholmod
# from sksparse.cholmod import cholesky
# import torch

class Solvers:
    def __init__(self) -> None:
        pass

    @classmethod
    def solveEqualityQP(self, H, g, A, b, type):
        """ Solve the equality constrained QP using some method """

        match type:
            case 'LU':
                K, r, m = self.EqualityQPKKT(H, g, A, b)
                x, lam = self.EqualityQPSolverLU(K, r, m)
            case 'LUSparse':
                K, r, m = self.EqualityQPKKT(H, g, A, b, sparse=True)
                x, lam = self.EqualityQPSolverLU(K, r, m, sparse=True)
            case 'LDL':
                K, r, m = self.EqualityQPKKT(H, g, A, b)
                x, lam = self.EqualityQPSolverLDL(K, r, m)
            case 'LDLSparse':
                K, r, m = self.EqualityQPKKT(H, g, A, b, sparse=True)
                x, lam = self.EqualityQPSolverLDL(K, r, m, sparse=True)
            case 'NullSpace':
                x, lam = self.EqualityQPSolverNullSpace(H, g, A, b)
            case 'RangeSpace':
                x, lam = self.EqualityQPSolverRangeSpace(H, g, A, b)
            case _:
                print("Solution type not defined")
                x, lam = 0, 0

        return x, lam

    @staticmethod
    def EqualityQPKKT(H, g, A, b, sparse=False):
        """ Setup the KKT system for an equality constrained QP"""
        # Initialize
        n = np.shape(H)[0]
        m = np.shape(A)[1]
        if not sparse:
            K = np.zeros((n + m, n + m), dtype=np.float64)
        else:
            K = scipy.sparse.lil_matrix((n + m, n + m), dtype=np.float64)
        r = np.zeros((n + m, 1))
        # Build KKT matrix and right-hand side vector
        K[0:n, 0:n] = H
        K[n:n+m, 0:n] = -A.T
        K[0:n, n:n+m] = -A
        r[0:n, 0:1] = -g
        r[n:n+m, 0:1] = -b
        return K, r, m
    
    @staticmethod
    def EqualityQPSolverLU(K, r, m, sparse=False):
        """ Solve system Ku = r using LU factorization """ 
        n = np.shape(K)[0] - m
        if not sparse:
                # Method 1
            lu, piv = scipy.linalg.lu_factor(K)
            u = scipy.linalg.lu_solve((lu, piv), r)
                # Method 2
            # L, U = scipy.linalg.lu(K, permute_l=True)
            # v = scipy.linalg.solve(L, r, lower=True)                # Solve:   PLv = r
            # u = scipy.linalg.solve_triangular(U, v, lower=False)    # Solve:   Uu = v
        else:
            Kfact = scipy.sparse.linalg.splu(K.tocsc())
            u = Kfact.solve(r)    
        # Map solution to design variables and Lagrange multipliers
        x = u[0:n]
        lam = u[n:n+m]
        return x, lam

    @classmethod
    def EqualityQPSolverLDL(self, K, r, m, sparse=False):
        """ Solve system Ku = r using LDL factorization """ 
        n = np.shape(K)[0] - m
        # Factorization:  PKP' = LDL' (Px = x(p))
        if not sparse:
                # Optimized implementation
            L, D, p = scipy.linalg.ldl(K, lower=True)           
            d = np.diagonal(D)
            z = scipy.linalg.solve_triangular(L, r, lower=True)      # Solve:   Lz = r
            v = (z.T/d).T                                            # Compute: v = z/d
            u = scipy.linalg.solve_triangular(L.T, v, lower=False)   # Solve:   L'u = v
                # Naive implementation
            # w = scipy.linalg.solve(L, r, lower=True)      # Solve:   Lw = r
            # v = scipy.linalg.solve(D, w)                  # Solve:   Dv = w
            # u = scipy.linalg.solve(L.T, v, lower=False)   # Solve:   L'u = v
        else:
            Kfact = sksparse.cholmod.cholesky(K.tocsc())
            u = Kfact.solve_A(r)

        # Map solution to design variables and Lagrange multipliers
        x = u[0:n]
        lam = u[n:n+m]
        return x, lam
    
    @staticmethod
    def EqualityQPSolverNullSpace(H, g, A, b):
        """ Solve system Ku = r using Null-Space procedure """ 
        n = np.shape(A)[0]
        m = np.shape(A)[1]
        Q, Rbar = scipy.linalg.qr(A)
        m1 = np.shape(Rbar)[1]
        Q1 = Q[:, 0:m1]
        Q2 = Q[:, m1:n]
        R = Rbar[0:m1, 0:m1]
        xY = scipy.linalg.solve(R.T, b)
        xZ = scipy.linalg.solve(Q2.T @ H @ Q2, -Q2.T @ (H @ Q1 @ xY + g))
        x = Q1 @ xY + Q2 @ xZ
        lam = scipy.linalg.solve(R, Q1.T @ (H @ x + g))
        return x, lam
    
    @staticmethod
    def EqualityQPSolverRangeSpace(H, g, A, b):
        """ Solve system Ku = r using Null-Space procedure """ 
        n = np.shape(A)[0]
        m = np.shape(A)[1]
        # 1) Cholesky factorize H
        c, L = scipy.linalg.cho_factor(H)
        # 2) Solve Hv = g for v
        v = scipy.linalg.cho_solve((c, L), g)
        # 3) Form H_A and its Cholesky factorization 
        H_A = A.T @ scipy.linalg.cho_solve((c, L), A)
        c_A, L_A = scipy.linalg.cho_factor(H_A)
        # 4) Solve for Lagrange multipliers
        lam = scipy.linalg.cho_solve((c_A, L_A), b + A.T @ v)       
        # 5) Solve for design variables
        x = scipy.linalg.cho_solve((c, L), A @ lam - g)
        return x, lam
    
    @classmethod
    def InequalityQPSolverPrimal(self, H, g, A, b, x0, W):

        # Settings
        maxiter = 10       # Maximum no. of iterations
        numtol = 1e-6       # Numerical tolerance for checks

        # Initialize
        n = np.shape(x0)[0]
        m = np.shape(A)[1]
        lams = np.zeros((m, 1))
        I = np.arange(m)
        xkStore = np.zeros((n, maxiter))
        WStore = np.zeros((m, maxiter))
        sol = {}
        sol["succes"] = 0

        # Set initial values
        xk = x0
        sol["x0"] = x0

        for k in range(maxiter):
            # Store values
            xkStore[:, k:k+1] = xk
            WStore[:, k] = np.isin(I, W)
            # Define system from the working set W
            Aw = A[:, W]
            zerow = np.zeros((len(W), 1))
            # Solve the equality constrained QP for the search direction pk
            gk = H @ xk + g
            pk, lamk = Solvers.solveEqualityQP(H, gk, Aw, zerow, 'LDL')    
            # print(f"Iteration: k = {k}")
            # print(f"xk = \n {xk}")
            # print(f"pk = \n {pk}")
            # print(f"lamk = \n {lamk}")
            # print(f"W = \n {W}")
            if np.isclose(np.linalg.norm(pk), 0):            
                if np.all(lamk >= numtol):           
                    # The optimal solution has been founds
                    xs = xk            
                    lams[W] = lamk
                    print(f"Solution found after {k} iterations: x = \n {xs}")
                    sol["succes"] = 1
                    # Store values
                    xkStore[:, k:k+1] = xk
                    WStore[:, k] = np.isin(I, W)
                    break
                else:
                    # Remove the most negative constraint from the working set
                    j = np.argmin(lamk)
                    W = np.setdiff1d(W, W[j])
                    # xk = xk                           
            else:    
                # Compute the distance to the nearest inactive constraint in the 
                # search direction pk

                # Extract system which is not contained in the working set
                nW = np.setdiff1d(I, W)             
                Anw = A[:, nW]               
                # Find blocking constraints
                blockCrit = Anw.T @ pk
                blockCrit = blockCrit.flatten() 
                blockCrit < 0
                iblock = nW[blockCrit < -numtol]
                # Find step length        
                alphas = (b[iblock] - Anw[:, iblock].T @ xk)/( Anw[:, iblock].T @ pk )
                alpha = np.minimum(1, alphas)
                j = np.argmin(alpha)
                alphak = alpha[j]
                if alphak < 1:      # Take reduced step and add blocking constraint
                    xk = xk + alphak*pk
                    W = np.append(W, j)
                    W.sort()
                else:               # Take full step
                    xk = xk + pk

        # Append results
        sol["iter"] = k
        sol["xiter"] = xkStore[:, 0:k+1]
        sol["Witer"] = WStore[:, 0:k+1]
        if sol["succes"] == 0:
            print("Solution could not be found")     

        return sol
    

