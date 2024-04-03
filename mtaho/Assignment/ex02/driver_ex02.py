import scipy

data = scipy.io.loadmat('/home/mtaho/Code/Courses/ConstrainedOptimization/mtaho/Assignment/ex02/QP_Test.mat')

# Load data
H = data['H']
g = data['g']
C = data['C']
dl = data['dl']
du = data['du']
l = data['l']
u = data['u']

print(data)