#------------------------------------------------------------------------------
# Modelling in Biology Python Training Exercise 2
#------------------------------------------------------------------------------

from distutils.spawn import find_executable
import matplotlib
import matplotlib.pyplot as plt
# Check if we can use pretty LATEX formatting in plots
if find_executable('latex') and matplotlib.checkdep_usetex(True):
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
import numpy as np
import os
# Use pretty Seaborn-style plots if installed
try:
    import seaborn as sns
    sns.set()
except:
    pass

# Create directory for results
result_dir = os.path.join(os.getcwd(),'results')
if not os.path.exists(result_dir):
    os.mkdir(result_dir)


#------------------------------------------------------------------------------
# Exercise 1: Euler's Method. k=0.25, h=0.01, x(0)=5
#------------------------------------------------------------------------------

# Parameters:
k=0.25
h1=0.01
initial = 5.

# We hold our arrays in lists (of arrays) to make plotting at different 
# h values easier
ts = []
xs = []
analyticals = []
# Define time span [0,10] with time step h1
ts.append(np.arange(0,10,h1))
xs.append(np.zeros(ts[0].shape))

# Set initial condition:
xs[0][0] = initial


# Euler's Method implementation:
for i in range(len(ts[0])-1):
    xs[0][i+1]=xs[0][i]+h1*-k*xs[0][i]  # will calculate xs[0](2:length(ts[0]))

# Analytical solution
analyticals.append(initial*np.exp(-k*ts[0]))  # analytical solution h=0.01

# Mean Squared Error
MSE1=np.mean((xs[0]-analyticals[0])**2)

print("MSE (h={}): {}".format(h1,MSE1))

#------------------------------------------------------------------------------
# Exercise 2: Euler's Method. k=0.25, h=0.001, x(0)=5
#------------------------------------------------------------------------------

# Parameters:

k=0.25
h2=0.001


#Define time span [0,10] with time step h2
ts.append(np.arange(0,10,h2))
xs.append(np.zeros(ts[1].shape))

# Set initial condition:
xs[1][0] = initial

for i in range(len(ts[1])-1):
    xs[1][i+1]=xs[1][i]+h2*-k*xs[1][i]  # will calculate xs[1](2:length(ts[0]))

# Analytical Solution

analyticals.append(initial*np.exp(-k*ts[1])) # analytical solution h=0.001

# Mean Squared Error
MSE2=np.mean((xs[1]-analyticals[1])**2)

print("MSE (h={}): {}".format(h2,MSE2))

f,a = plt.subplots(nrows=2,figsize=(8,6))
hs = [h1,h2]
for i in range(2):
    a[i].plot(ts[i],xs[i],'b')
    a[i].plot(ts[i],analyticals[i],'r--')
    a[i].set_xlabel('Time')
    a[i].set_ylabel('x')
    a[i].set_title('k = 0.25, h = {}'.format(hs[i]))
    
f.tight_layout()
f.savefig(os.path.join(result_dir,'training2.pdf'))
f.savefig(os.path.join(result_dir,'training2.svg'))