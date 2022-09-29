#------------------------------------------------------------------------------
# Modelling in Biology Python Training Exercise 5
#------------------------------------------------------------------------------

from distutils.spawn import find_executable
# We need functools.partial to set parameter values in functions
from functools import partial
# We must import matplotlib.pyplot to allow us to plot our solutions
import matplotlib
import matplotlib.pyplot as plt
# Check if we can use pretty LATEX formatting in plots
if find_executable('latex') and matplotlib.checkdep_usetex(True):
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
# We use numpy to give us access to advanced mathematics functions
import numpy as np
import os
# We need to import the ODE solver
from scipy.integrate import solve_ivp
# We need the fsolve here too
from scipy.optimize import fsolve
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


def odefun(t,y,eta):
    # dydt = y[1] (i.e. we capture the second order behaviour in the second
    # element of the vector Y = [y, dydt])
    # d2ydt2 = -y - eta*dydt
    return [y[1], -y[0]-eta*y[1]]

# Define parameters
etas = [0, 0.03, 7] # Parameter eta vector
y0 = [2, 10] # Initial condition

# Initialize variables for calling ODE solver
tspan = [0, 100]
# Use high-resolution t_eval for pretty traces
t_eval = np.linspace(*tspan,1000)

ys = []
dys = []

# Call ODE solver for each of the 3 values of the parameter eta
for eta in etas:
    soln = solve_ivp(partial(odefun,
                             eta=eta), 
                     tspan, 
                     y0, 
                     t_eval=t_eval,
                     max_step=0.5,
                     rtol=1e-3,
                     atol=1e-6)
    # Store the output in the two vectors y and ydot
    ys.append(soln.y[0,:])
    dys.append(soln.y[1,:])


f,a = plt.subplots(ncols=2,figsize=[14,7])
for i in range(len(etas)):
    # Plot overlaid trajectories y(t) as a function of time.
    # for each value of parameter eta.
    a[0].plot(soln.t,ys[i])
    # Plot overlaid phase plane for each value of parameter eta.
    a[1].plot(ys[i],dys[i])
a[0].set_xlabel('$t$')
a[0].set_ylabel('$y(t)$')
a[0].set_title('y(t) vs. t')
a[0].legend(['$\eta=0$','$\eta=0.003$','$\eta=7$'])

a[1].set_xlabel('$y(t)$')
a[1].set_ylabel('$\dot{y}(t)$')
a[1].legend(['$\eta=0$','$\eta=0.003$','$\eta=7$'])
a[1].set_title('Phase plane')
f.tight_layout()
f.savefig(os.path.join(result_dir,'training5.pdf'))
f.savefig(os.path.join(result_dir,'training5.svg'))