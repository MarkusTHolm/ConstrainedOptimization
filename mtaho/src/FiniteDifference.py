import numpy as np
import matplotlib.pyplot as plt
from cycler import cycler

from mtaho.plot_settings import define_plot_settings

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
            h = pert*np.max(np.array([1.0, np.abs(x[i])]))
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
            h = pert*np.max(np.array([1.0, np.abs(x[i])]))
            xh = x.copy()
            xh[i] = xh[i] + h
            h = xh[i] - x[i]
            ch = fun(xh)[1]
            dcdxi = (ch - c)/h
            J[:, i] = dcdxi.T
        return J
    
    def plotFiniteDifferenceCheck(fun, x0, f0, df0, d2f0, 
                                  outPath=None, fontSize=20):
        """ Check sensitivies with finite difference values """

        # Dimension of the problem
        n = len(x0)

        # Plot FD check of gradients
        pertRange = np.logspace(-12, -1, 100)
        relErrGrad = np.zeros((n, len(pertRange)))
        relErrHess = np.zeros((n, n, len(pertRange)))
        for i, pert in enumerate(pertRange):
            g = FD.gradient(fun, f0, x0, pert)
            H = FD.jacobian(fun, df0, x0, pert)
            relErrGrad[:, i:i+1] = (g - df0)/df0
            relErrHess[0:n, 0:n, i] = np.abs((H-d2f0)/d2f0)

        # Plot finite difference check
        define_plot_settings(fontSize)
        fig, axs = plt.subplots(1, 2, figsize=(13, 5))
        custom_cycler = (cycler(color=plt.rcParams['axes.prop_cycle'].by_key()['color']) +
                         cycler(linestyle=['solid', 'dashed', 'dashdot',
                                           'solid', 'dashed', 'dashdot',
                                           'solid', 'dashed', 'dashdot',
                                           'solid']))
        for ax in axs:
            ax.set_prop_cycle(custom_cycler)
        
        axs[0].loglog(pertRange, relErrGrad.T, 
                      label=[r'$\nabla f_{('+f'{i}'+')}$' for i in range(n)])
        for i in range(n):
            for j in range(n):
                axs[1].loglog(pertRange, relErrHess[i, j, :], 
                              label='$H_{('+f'{i}, {j}'+')}$')
        axs[0].set_title('Gradient FD check')
        axs[1].set_title('Hessian FD check')
        for ax in axs:
            ax.set_xlabel("Pertubation constant")
            ax.set_ylabel("Relative error")
            ax.legend(bbox_to_anchor=(1.05, 1), fontsize=fontSize-4)

        fig.tight_layout()

        if not outPath == None:
            fig.savefig(outPath)
