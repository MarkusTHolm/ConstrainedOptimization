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
                    fontSize = 20, figSize = (7, 5)):

        x = np.linspace(xlim[0], xlim[1], nPoints)
        y = np.linspace(ylim[0], ylim[1], nPoints)
        X, Y = np.meshgrid(x, y)
        Z = fun(X, Y)

        # Plot results
        define_plot_settings(fontSize)

        # Contour plot
        fig, ax = plt.subplots(figsize = figSize )
        cs = ax.contourf(X, Y, Z, 200,\
                         norm=LogNorm())#locator=ticker.LogLocator())
        ax.contour(X, Y, Z,\
                   locator=ticker.LogLocator(),
                   linewidths=.25,
                   colors='w')
        cbar = fig.colorbar(cs)
        ax.set_xlabel('$x_1$')
        ax.set_ylabel('$x_2$')
        # ax.grid()
        fig.tight_layout()
        
        if not outPath == None:
            fig.savefig(outPath)