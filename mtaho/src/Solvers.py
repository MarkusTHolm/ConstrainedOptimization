import numpy as np
import scipy
import cvxopt
import cvxopt.cholmod

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
            K = np.zeros((n + m, n + m))
        else:
            K = scipy.sparse.lil_matrix((n + m, n + m))
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
        # Solve system using LDL factorization: 
        # u = np.linalg.solve(K, r)
        if not sparse:
            lu, piv = scipy.linalg.lu_factor(K)
            u = scipy.linalg.lu_solve((lu, piv), r)
        else:
            Kfact = scipy.sparse.linalg.splu(K.tocsc())
            u = Kfact.solve(r)    
        # Map solution to design variables and Lagrange multipliers
        x = u[0:n]
        lam = u[n:n+m]
        return x, lam

    @staticmethod
    def EqualityQPSolverLDL(K, r, m, sparse=False):
        """ Solve system Ku = r using LDL factorization """ 
        n = np.shape(K)[0] - m
        # Factorization:  PKP' = LDL' (Px = x(p))
        if not sparse:
            L, D, p = scipy.linalg.ldl(K, lower=True)           
            w = np.linalg.solve(L, r)      # Solve:   Lw = r
            v = np.linalg.solve(D, w)      # Solve:   Dv = w
            u = np.linalg.solve(L.T, v)    # Solve:   L'u = y
        else:
            # L, D, p = scipy.linalg.ldl(K, lower=True)           
            # w = np.linalg.solve(L, r)      # Solve:   Lw = r
            # v = np.linalg.solve(D, w)      # Solve:   Dv = w
            # u = np.linalg.solve(L.T, v)    # Solve:   L'u = y

            K = K.tocoo()
            K = cvxopt.spmatrix(K.data, K.row.astype(np.int64),
                                        K.col.astype(np.int64))
            # B = cvxopt.matrix(r)
            # cvxopt.cholmod.numeric(K, F)
            # cvxopt.cholmod.spsolve(K, B, sys=1)
            # u = np.array(B)[:, 0]
            A = K
            B = cvxopt.matrix(r)
        
            # X = cvxopt.cholmod.linsolve(A, B)

            F = cvxopt.cholmod.symbolic(A)
            cvxopt.cholmod.numeric(A, F)
            cvxopt.cholmod.solve(F, B, sys=1)


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