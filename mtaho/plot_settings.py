import matplotlib
import matplotlib.pyplot as plt


def define_plot_settings(font_size = 20):
        font = {'family' : 'Times New Roman',
                'size'   : font_size}
        matplotlib.rc('font', **font)
        SMALL_SIZE =  font_size - 2
        MEDIUM_SIZE = font_size
        BIGGER_SIZE = font_size + 2
        plt.rc('font', size=SMALL_SIZE)          # Controls default text sizes
        plt.rc('axes', titlesize=SMALL_SIZE)     # Font size of the axes title
        plt.rc('axes', labelsize=MEDIUM_SIZE)    # Font size of the x and y labels
        plt.rc('xtick', labelsize=SMALL_SIZE)    # Font size of the tick labels
        plt.rc('ytick', labelsize=SMALL_SIZE)    # Font size of the tick labels
        plt.rc('legend', fontsize=SMALL_SIZE)    # Legend font size
        plt.rc('figure', titlesize=BIGGER_SIZE)  # Fontsize of the figure title
        plt.rcParams.update({
                "text.usetex": True,
                'text.latex.preamble': r'\usepackage{amsmath}'
                                #        r'\usepackage{commath}',
        })