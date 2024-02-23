import matplotlib.pyplot as plt
import numpy as np
import os
import sys

# project_dir = r"C:\Users\marku\OneDrive - Danmarks Tekniske Universitet"+\
#               r"\PhD\08_Courses\ConstrainedOptimization\Code"
projectDir = "/home/mtaho/Code/Courses/ConstrainedOptimization"
sys.path.append(os.path.realpath(f"{projectDir}"))

from mtaho.plot_settings import define_plot_settings
from mtaho.src.FiniteDifference import FD
from mtaho.src.BasicOpt import Opt

workDir = f"{projectDir}/mtaho/w01"

# 1) Contour plot
def fContour(X, Y):
    p = [100, 1]
    f = p[0]*(Y - X**2)**2 + p[1]*(1 - X)**2
    return f

Opt.contourPlot(fContour, f'{workDir}/ex2Contour.png',
                xlim=[-2, 2], ylim=[-1, 3],colorScale='log')

# 4) Analytical gradient and Hessian function
def funEx2(x):
    p = [100, 1]
    f = p[0]*(x[1] - x[0]**2)**2 + p[1]*(1 - x[1])**2
    df = np.array([[-4*p[0]*(x[1] - x[0]**2)*x[0]               ],
                   [2*p[0]*(x[1] - x[0]**2) - 2*p[1]*(1 - x[1]) ]])
    d2f = np.array([[4*p[0]*(3*x[0]**2 - x[1])  ,-4*p[0]*x[0]          ],
                    [-4*p[0]*x[0]               ,2*(p[0] + p[1])  ]])
    return f, df, d2f

x0 = np.array([2.0, 3.0])
f0, df0, d2f0 = funEx2(x0)              

# Gradient
df0_num = FD.gradient(funEx2, f0, x0)

print(f"df0 = {df0.T}, \n df0_num = {df0_num.T}")
print(f"Relative difference: (df0_num-df0)/df0 = \n "
      f"{(df0_num-df0)/df0*100} %")

# Jacobian
d2f0_num = FD.jacobian(funEx2, df0, x0)

print(f"d2f0 = \n {d2f0}, \n d2f0_num = \n {d2f0_num.T}")
print(f"Relative difference: (d2f0_num-d2f0)/d2f0 = \n"
      f"{(d2f0_num-d2f0)/d2f0*100} %")

FD.plotFiniteDifferenceCheck(funEx2, x0, f0, df0, d2f0,
                             outPath=f'{workDir}/ex2FDcheck.png')

# plt.show()