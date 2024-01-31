import numpy as np
import matplotlib.pyplot as plt

from mtaho.plot_settings import define_plot_settings

class Opt:
    def __init__(self) -> None:
        pass

    def contourPlot(fun, outPath, xlim=[-10, 10], ylim=[-10, 10], nPoints=200):

        x = np.linspace(xlim[0], xlim[1], nPoints)
        y = np.linspace(ylim[0], ylim[1], nPoints)
        X, Y = np.meshgrid(x, y)
        Z = fun(X, Y)

        # Plot results
        define_plot_settings(16)

        # Contour plot
        fig, ax = plt.subplots()
        cs = ax.contourf(X, Y, Z)
        cbar = fig.colorbar(cs)
        ax.grid()
        fig.savefig(outPath)