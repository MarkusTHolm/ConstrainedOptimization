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



# 3) Analytical gradient and Hessian function
def funEx3(x):
    d2c = np.zeros((2, 2, 2))
    c = np.array([[np.exp(x[0]) - x[1]  ],
                  [x[0]**2 - 2*x[1]     ]])
    dc = np.array([[np.exp(x[0])    ,2*x[0] ],
                   [-1              ,-2     ]])
    d2c[:, :, 0] = np.array([[np.exp(x[0])  , 0 ],
                             [0             , 0 ]])
    d2c[:, :, 1] = np.array([[2             , 0 ],
                             [0             , 0 ]])
    return c, dc, d2c

# 4) Analytical function value and Jacobian
def funJacEx3(x):
    tmp = []
    d2c = np.zeros((2, 2, 2))
    c = np.array([[np.exp(x[0]) - x[1]  ],
                  [x[0]**2 - 2*x[1]     ]])
    dc = np.array([[np.exp(x[0])    ,2*x[0] ],
                   [-1              ,-2     ]])
    J = dc.T
    return tmp, c, J

# 5) Finite difference check in x0
x0 = np.array([1.7, 2.1])
tmp, c0, J0 = funJacEx3(x0)              

# # Jacobian
J0_num = FD.jacobian(funJacEx3, c0, x0)

print(f"J0 = \n {J0.T}, \nJ0_num = \n {J0_num.T}")
print(f"Relative difference: (df0_num-df0)/df0 = \n "
      f"{(J0_num-J0)/J0*100} %")

# Hessian
c0, dc0, d2c0 = funEx3(x0)

d2c0_num = FD.hessian(funEx3, dc0, x0)
    
print("hello")
print(f"d2c0 = \n {d2c0.T}, \n d2f0_num = \n {d2c0_num.T}")
print(f"Relative difference: (d2f0_num-d2f0)/d2f0 = \n"
      f"{(d2c0_num-d2c0)/d2c0*100} %")

# FD.plotFiniteDifferenceCheck(funEx2, x0, f0, df0, d2f0,
#                              outPath=f'{workDir}/ex2FDcheck.png')

# plt.show()