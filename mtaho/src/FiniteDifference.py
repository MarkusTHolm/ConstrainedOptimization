import numpy as np
import matplotlib.pyplot as plt

class FD:
    def __init__(self) -> None:
        pass

    def gradient(fun, f, x, pert=None):
        """ Approximate the gradient using forward finite-difference """
        eps = np.finfo(np.float64).eps
        if not pert:
            pert = np.sqrt(eps)
        nx = len(x)

        g = np.zeros((nx, 1))
        for i in range(nx):
            h = pert*np.max(np.array(1.0, np.abs(x[i])))
            xh = x.copy()
            xh[i] = xh[i] + h
            h = xh[i] - x[i]
            fh = fun(xh)[0]
            g[i] = (fh - f)/h
        return g

    def jacobian(fun, c, x, pert=None):
        """ Approximate the gradient using forward finite-difference """
        eps = np.finfo(np.float64).eps
        if not pert:
            pert = np.sqrt(eps)
        nx = len(x)
        nc = len(c)

        J = np.zeros((nc, nx))
        for i in range(nx):
            h = pert*np.max(np.array(1.0, np.abs(x[i])))
            xh = x.copy()
            xh[i] = xh[i] + h
            h = xh[i] - x[i]
            ch = fun(xh)[1]
            dcdxi = (ch - c)/h
            J[:, i] = dcdxi.T
        return J
    
    def plotFiniteDifferenceCheck(fun, x0, f0, df0, d2f0, outPath=None):

        # Plot FD check of gradients
        pertRange = np.logspace(-12, -1, 100)
        relErrGrad = np.zeros((2, len(pertRange)))
        relErrHess = np.zeros((2, 2, len(pertRange)))
        for i, pert in enumerate(pertRange):
            g = FD.gradient(fun, f0, x0, pert)
            H = FD.jacobian(fun, df0, x0, pert)
            relErrGrad[:, i:i+1] = (g - df0)/df0
            relErrHess[0:2, 0:2, i] = np.abs((H-d2f0)/d2f0)

        # Finite difference check
        fig, axs = plt.subplots(1, 2, figsize=(10, 4))
        axs[0].loglog(pertRange, relErrGrad.T)
        axs[1].loglog(pertRange, relErrHess[1, 1, :])
        axs[0].set_title('Gradient FD check')
        axs[1].set_title('Hessian FD check')
        for i in range(len(axs)):
            axs[i].set_xlabel("Pertubation constant: ")
            axs[i].set_ylabel("Relative error")
        fig.tight_layout()

        if not outPath == None:
            fig.savefig(outPath)
