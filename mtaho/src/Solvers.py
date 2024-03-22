import numpy as np
import scipy
import scipy.sparse as sp
import cvxopt
import cvxopt.cholmod
import sksparse
import sksparse.cholmod
from sksparse.cholmod import cholesky
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
    def QPSolverInequalityActiveSet(self, H, g, A, b, x0, W=[]):
        """ Solve an inequality constrained convex QP using the 
        primal active-set method for a feasible starting point x0,
        (and optionally corresponding active set).
        
        Problem:
            min_x : 0.5*x'Hx + g'x
            s.t.  :  A'x =  b
                  :  C'x >= d
        """

        # Initialize
        n = np.shape(x0)[0]
        m = np.shape(A)[1]
        printValues = n < 10
        lams = np.zeros((m, 1))
        I = np.arange(m)

        # Settings
        maxiter = 100*(n+m)    # Maximum no. of iterations
        numtol = 1e-9          # Numerical tolerance for checks

        # Set initial values
        xk = x0

        # Store solution info
        xkStore = np.zeros((n, maxiter))
        WStore = np.zeros((m, maxiter))
        sol = {}
        sol["x0"] = x0
        sol["succes"] = 0
        
        for k in range(maxiter):
            # Store values
            xkStore[:, k:k+1] = xk
            WStore[:, k] = np.isin(I, W)
            # Define system from the working set W
            Aw = A[:, W]
            zerow = np.zeros((len(W), 1))
            # Solve the equality constrained QP for the search direction pk
            gk = H @ xk + g
            pk, lamk = Solvers.solveEqualityQP(H, gk, Aw, zerow, 'LUSparse') 
            if printValues:   
                print(f"Iteration: k = {k}")
                print(f"xk = \n {xk}")
                print(f"pk = \n {pk}")
                print(f"lamk = \n {lamk}")
                print(f"W = \n {W}")
            if np.isclose(np.linalg.norm(pk), 0):            
                if np.all(lamk >= numtol):           
                    # The optimal solution has been founds     
                    lams[W] = lamk
                    sol["succes"] = 1
                    # Store values
                    xkStore[:, k:k+1] = xk
                    WStore[:, k] = np.isin(I, W)
                    break
                else:
                    # Remove the most negative constraint from the working set
                    j = np.argmin(lamk)
                    W = np.setdiff1d(W, W[j])
                    W = W.tolist()
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
                blockMask = blockCrit < -numtol
                iblock = nW[blockMask]                
                # Find step length        
                cnW = (Anw[:, blockMask].T @ xk + b[iblock])
                alphas = -cnW/( Anw[:, blockMask].T @ pk )
                alpha = np.minimum(1, alphas)
                j = np.argmin(alpha)
                alphak = alpha[j]
                if alphak < 1:      # Take reduced step and add blocking constraint
                    xk = xk + alphak*pk
                    W.append(iblock[j])
                    W.sort()
                else:               # Take full step
                    xk = xk + pk

        # Append results
        sol["iter"] = k
        sol["xiter"] = xkStore[:, 0:k+1]
        sol["Witer"] = WStore[:, 0:k+1]
        if sol["succes"] == 0:
            print("Solution could not be found")
        elif sol["succes"] == 1:
            sol['x'] = xk
            print(f"Solution found after {k} iterations")

        return sol
    
    @classmethod
    def QPSolverInteriorPoint(self, H, g, C, d, A=None, b=None, x0=None,
                              loud=False):
        """ 
        Solve a convex QP using the primal-dual interior point
        algorithm 
        Problem:
            min_x   : 0.5*x'Hx + g'x
            s.t.    : A'x = b
                    : C'x >= d 
        """
        # Settings
        maxiter = 100      # Maximum no. of iterations
        tolL = 1e-9        # Lagrangian gradient
        tolA = 1e-9        # Equality constraint
        tolC = 1e-9        # Inequality constraint
        tolmu = 1e-9       # Dual gap
        eta = 0.995        # Damping parameter for step length 

        # Initialize
        n = np.shape(x0)[0]
        if A is not None:
            me = np.shape(A)[1]
            yBar = np.zeros((me, 1))
        else:
            me = 0
            A = np.zeros((n, 1))
            b = np.zeros((1, 1))
            yBar = np.zeros((1, 1))
        mi = np.shape(C)[1]
        xBar = x0.copy()        
        zBar = np.ones((mi, 1))
        sBar = np.ones((mi, 1))
        s = np.ones((mi, 1))
        z = np.ones((mi, 1))
        eVec = np.ones((mi, 1))
        rhs = np.zeros((n+me, 1))
        sol = {}
        sol["succes"] = 0
        xStore = np.zeros((n, maxiter))

        def computeResiduals(H, g, C, d, A, b, x, y, z, s):
            rL = H@x + g - A@y - C@z
            rA = b - A.T@x
            rC = s + d - C.T@x
            rSZ = s*z
            return rL, rA, rC, rSZ
        
        def factorize(H, C, z, s):  
            HBar = H + C@((z/s)*C.T)
            if me > 0:
                K = scipy.sparse.lil_matrix((n + me, n + me), dtype=np.float64)
                K[0:n, 0:n] = HBar
                K[n:n+me, 0:n] = -A.T
                K[0:n, n:n+me] = -A
                Kfact = sp.linalg.splu(sp.csc_matrix(K))  
            else:
                Kfact = sp.linalg.splu(sp.csc_matrix(HBar))                 
            return Kfact        
        
        def searchDirection(Kfact, rL, rA, z, s, C, rC, rSZ):
            rLBar = rL - (C*(z/s).T)@(rC - rSZ/z)            
            rhs[0:n] = -rLBar
            rhs[n:n+me] = -rA
            u = Kfact.solve(rhs)    
            dx = u[0:n]
            if me > 0:
                dy = u[n:(n+me)]
            else: 
                dy = 0.0
            dz = -(z/s)*C.T@dx + (z/s)*(rC - rSZ/z)
            ds = -rSZ/z - (s/z)*dz
            return dx, dy, dz, ds
        
        def feasibleStepLength(z, dz, s, ds):
            # Compute the largest alphaAff that retains a feasible step
            alpha_z = np.min(np.minimum(1, -z[dz < 0]/dz[dz < 0]))
            alpha_s = np.min(np.minimum(1, -s[ds < 0]/ds[ds < 0]))
            alpha = np.min([alpha_z, alpha_s])
            return alpha
        
        # Find a suitable initial point 

        rL, rA, rC, rSZ = computeResiduals(H, g, C, d, A, b,
                                           xBar, yBar, zBar, sBar)
        Kfact = factorize(H, C, z, s)
        dxAff, dyAff, dzAff, dsAff = searchDirection(Kfact, rL, rA, z, s, C, rC, rSZ)
        x = xBar
        y = yBar
        z = np.maximum(1, np.abs(zBar + dzAff))
        s = np.maximum(1, np.abs(sBar + dsAff))

        # Primal-dual predictor corrector interior point algorithm

        rL, rA, rC, rSZ = computeResiduals(H, g, C, d, A, b,
                                           x, y, z, s)
        mu = (z.T@s)/mi
        for k in range(maxiter):
            Kfact = factorize(H, C, z, s)
            # Compute affine direction
            dxAff, dyAff, dzAff, dsAff = searchDirection(Kfact, rL, rA, z, s, C, rC, rSZ)
            alphaAff = feasibleStepLength(z, dzAff, s, dsAff)
            # Compute the duality gap and centering parameter
            muAff = ((z + alphaAff*dzAff).T@(s + alphaAff*dsAff))/mi
            sigma = (muAff/mu)**3
            # Compute affine centering-correction direction
            rSZBar = rSZ + dsAff*dzAff - sigma*mu*eVec
            dx, dy, dz, ds = searchDirection(Kfact, rL, rA, z, s, C, rC, rSZBar)
            alpha = feasibleStepLength(z, dz, s, ds)
            alphaBar = eta*alpha
            # Update iteration
            x += alphaBar*dx
            y += alphaBar*dy
            z += alphaBar*dz
            s += alphaBar*ds
            rL, rA, rC, rSZ = computeResiduals(H, g, C, d, A, b,
                                               x, y, z, s)
            mu = (z.T@s)/mi
            # Check stopping criteria
            if (np.linalg.norm(rL, np.inf) < tolL) and \
               (np.linalg.norm(rA, np.inf) < tolA) and \
               (np.linalg.norm(rC, np.inf) < tolC) and \
               (np.linalg.norm(mu, np.inf) < tolmu):
                sol["succes"] = 1
                break
            # Print and store results
            if loud:
                print(f"k = {k}: "
                    f"rL = {np.linalg.norm(rL, np.inf):1.1e}, "
                    f"rA = {np.linalg.norm(rA, np.inf):1.1e}, "
                    f"rC = {np.linalg.norm(rC, np.inf):1.1e}, "
                    f"mu = {np.linalg.norm(mu, np.inf):1.1e}")
            xStore[:, k:k+1] = x
        
        # Append results
        sol["iter"] = k
        sol["xiter"] = xStore[:, 0:k]
        sol["x"] = x
        if sol["succes"] == 0:
            print("Solution could not be found")    
        elif sol["succes"] == 1:
            print(f"Solution found after {k} iterations")

        return sol

    @classmethod
    def LPSolverRevisedSimplex(self, g, A, b, x0):
        """ Solve a LP in standard form using the revised simplex method """

        # Settings
        maxiter = 100       # Maximum no. of iterations
        numtol = 1e-6       # Numerical tolerance for checks

        # Initialize
        n = np.shape(x0)[0]
        m = np.shape(A)[0]
        printValues = n < 10
        x = x0.copy()                         # Start-iterate
        Is = np.arange(n)                     # Full index set
        active = np.isclose(x[:,0], 0)        # Find non-basic variables
        Ns = Is[active]                       # Non-basic set
        Bs = np.setdiff1d(Is, Ns)             # Basic set

        sol = {}
        sol['succes'] = 0
        xkStore = np.zeros((n, maxiter))

        for k in range(maxiter):
            # Store values
            xkStore[:, k:k+1] = x

            # Define non-basic and variables
            N = A[:, Ns]
            B = A[:, Bs]
            xN = x[Ns]
            xB = x[Bs]     
            gN = g[Ns]
            gB = g[Bs]

            # Solve for lagrange multiplers for the inequality constraints
            mu = np.linalg.solve(B.T, gB)
            
            # Find lagrange multipliers for the bound constraints
            lam = gN - N.T @ mu

            if printValues:
                print(f"--- Iteration {k}: ---"
                  f"\n x=\n{x}, \n Bs={Bs}, \n Ns={Ns}, \n mu=\n{mu}, \n lam=\n{lam}")

            # Find leaving index
            s = np.argmin(lam)

            # Check for solution (all inequality lagrange multipliers non-negative)
            if lam[s] >= 0:
                print(f"*** Solution found after {k} iterations")
                if printValues:
                    print(f"x = \n {x}")
                sol["succes"] = 1
                sol['x'] = x
                break
            else:
                
                # Find global index that corresponds to index s
                i_s = Ns[s]         

                # Check for unbounded problem
                h = np.linalg.solve(B, A[:, i_s:i_s+1])  # TODO: Re-use the factorization of B' 
                hpos = h > 0

                if np.sum(hpos) == 0:
                    print(f"Error: Problem is unbounded, i.e. has no solution")
                else:
                    # Find smallest step direction
                    j = np.argmin(xB[hpos]/h[hpos].T)
                    
                    # Compute step length
                    alpha = xB[j]/h[j]
                    
                    # Increment non-basic and basic variables
                    xB -= alpha*h
                    xN[:] = 0
                    xN[s] = alpha

                    # Update x
                    x[Ns] = xN
                    x[Bs] = xB

                    # Update basic and non-basic sets with leaving and entering indices
                    i_j = Bs[j]
                    Bs = np.union1d(np.setdiff1d(Bs, i_j), i_s)
                    Ns = np.union1d(np.setdiff1d(Ns, i_s), i_j)

        # Append results
        sol["iter"] = k
        sol["xiter"] = xkStore[:, 0:k+1]
        if sol["succes"] == 0:
            print("Solution could not be found") 

        return sol
       



