import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, ticker
from matplotlib.colors import LogNorm

from mtaho.plot_settings import define_plot_settings

class Opt:
    def __init__(self) -> None:
        pass

    def contourPlot(fun, outPath=None,
                    xlim=[-10, 10], ylim=[-10, 10], nPoints=400,
                    fontSize = 20, figSize = (7, 5), 
                    colorScale='linear', vmax=None):

        x = np.linspace(xlim[0], xlim[1], nPoints)
        y = np.linspace(ylim[0], ylim[1], nPoints)
        X, Y = np.meshgrid(x, y)
        Z = fun(X, Y)

        if colorScale == 'linear':
            norm = 'linear'
            nlevels = 20
            if vmax is not None:
                levels = np.arange(0, vmax, vmax/nlevels)
            else:
                levels = np.arange(np.min(Z), np.max(Z)*1.1, np.max(Z)/nlevels)
            locator=ticker.LinearLocator()
        elif colorScale == 'log':
            norm = LogNorm()
            locator = ticker.LogLocator()
            levels = 1

        # Plot results
        define_plot_settings(fontSize)

        # Contour plot
        fig, ax = plt.subplots(figsize = figSize)
        cs = ax.contourf(X, Y, Z,
                         levels=levels,
                         norm=norm)
        ax.contour(X, Y, Z,\
                   levels = levels,
                   locator=locator,
                   linewidths=.25,
                   colors='w')
        
        cbar = fig.colorbar(cs)

        ax.set_xlabel('$x_1$')
        ax.set_ylabel('$x_2$')
        fig.tight_layout()
        
        if not outPath == None:
            fig.savefig(outPath)

        return fig, ax