import matplotlib.pyplot as plt
import numpy as np
import os
import sys

# project_dir = r"C:\Users\marku\OneDrive - Danmarks Tekniske Universitet"+\
#               r"\PhD\08_Courses\ConstrainedOptimization\Code"
projectDir = "/home/mtaho/OneDrive/PhD/08_Courses/ConstrainedOptimization/Code"
sys.path.append(os.path.realpath(f"{projectDir}"))

from mtaho.plot_settings import define_plot_settings
from mtaho.src.FiniteDifference import FD
from mtaho.src.BasicOpt import Opt

workDir = f"{projectDir}/mtaho/w01"

# 1) Contour plot
def fContour(X, Y):
    f = X**2 - 2*X + 3*X*Y + 4*Y**3
    return f

Opt.contourPlot(fContour, f'{workDir}/ex1Contour.png')

# 5) Finite difference check
def funEx1(x):
    f = x[0]**2 - 2*x[0] + 3*x[0]*x[1] + 4*x[1]**3
    df = np.array([[2*x[0] - 2 + 3*x[1]],
                   [3*x[0] + 12*x[1]**2]])
    d2f = np.array([[2, 3],
                    [3, 24*x[1]]])
    return f, df, d2f

x0 = np.array([2.0, 3.0])
f0, df0, d2f0 = funEx1(x0)              

# Gradient
df0_num = FD.gradient(funEx1, f0, x0)

print(f"df0 = {df0.T}, \n df0_num = {df0_num.T}")
print(f"Relative difference: (df0_num-df0)/df0 = \n "
      f"{(df0_num-df0)/df0*100} %")

# Jacobian
d2f0_num = FD.jacobian(funEx1, df0, x0)

print(f"d2f0 = \n {d2f0}, \n d2f0_num = \n {d2f0_num.T}")
print(f"Relative difference: (d2f0_num-d2f0)/d2f0 = \n"
      f"{(d2f0_num-d2f0)/d2f0*100} %")

FD.plotFiniteDifferenceCheck(funEx1, x0, f0, df0, d2f0,
                             outPath=f'{workDir}/ex1FDcheck.png')

plt.show()