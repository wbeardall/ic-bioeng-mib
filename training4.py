#------------------------------------------------------------------------------
# Modelling in Biology Python Training Exercise 4
#------------------------------------------------------------------------------

# We need functools.partial to set parameter values in functions
from distutils.spawn import find_executable
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


# The toggle switch ODE function
def toggleswitch(t,x,alpha,beta,gamma):
    return [
        alpha/(1 + x[1]**4) - gamma*x[0],
        beta/(1+x[0]**4) - gamma*x[1]
        ]

def toggle_nullclines(x,alpha,beta,gamma):
    return [
        alpha/(1 + x[1]**4) - gamma*x[0],
        beta/(1+x[0]**4) - gamma*x[1]
        ]

#------------------------------------------------------------------------------
# Exercise 1: Plotting time courses
#------------------------------------------------------------------------------

# Define parameters
alpha  = 1
beta  = 1
gamma  = 0.3
x0 = [1, 2] # Any initial condition

# Call ODE solver
tspan = [0,200]
# Use high-resolution t_eval for pretty traces
t_eval = np.linspace(*tspan,1000)
soln = solve_ivp(partial(toggleswitch,
                         alpha=alpha,
                         beta=beta,
                         gamma=gamma),
                 tspan,x0,t_eval=t_eval)

f,a = plt.subplots(nrows=2,ncols=3,figsize=[15,10])
# Reshape axes array into 1d
a = np.reshape(a,-1)
for trace in soln.y:
    a[0].plot(soln.t,trace,linewidth=2)
a[0].set_title(r'1. $\alpha$ = $\beta$ = 1, $\gamma$ = 0.3')
a[0].set_xlabel('Time')
a[0].set_ylabel('Concentrations x and y')

# Compute stationary points
x0s = np.asarray([[1,1], [0,3], [3,0]])
for i in range(len(x0s)):
    sp = fsolve(partial(toggle_nullclines,
                        alpha=alpha,
                        beta=beta,
                        gamma=gamma),
                x0s[i])
    print("Fixed point {0} at x={1:.3f}, y={2:.3f}".format(i,sp[0],sp[1]))

#------------------------------------------------------------------------------
# Exercise 2: Calculating nullclines
#------------------------------------------------------------------------------

#Compute nullclines
yp1 = np.arange(0,10,0.01)
xp1 = alpha/gamma/(1 + yp1**4)
xp2 = np.arange(0,10,0.01)
yp2 = beta/gamma/(1 + xp2**4)

a[1].plot(xp1,yp1,c='r',linewidth=2)
a[1].plot(xp2,yp2,c='b',linewidth=2)
a[1].set_xlabel('Concentration x')
a[1].set_ylabel('Concentration y')
a[1].set_title("Nullclines")

#------------------------------------------------------------------------------
# Exercise 3: Plotting trajectories
#------------------------------------------------------------------------------

def plot_nullclines_and_trajectories(axis,x0s,alpha,beta,gamma):
    yp1 = np.arange(0,10,0.01)
    xp1 = alpha/gamma/(1 + yp1**4)
    xp2 = np.arange(0,10,0.01)
    yp2 = beta/gamma/(1 + xp2**4)
    # Plot nullclines, overlaid with phase plane
    axis.plot(xp1,yp1,c='k',linewidth=2)
    axis.plot(xp2,yp2,c='k',linewidth=2)
    axis.set_xlabel('Concentration x')
    axis.set_ylabel('Concentration y')
    axis.set_title(r'Phase plane trajectories $\alpha$ = {}, $\beta$ = {}, $\gamma$ = {}'.format(alpha,beta,gamma),wrap=True)
    
    
    for x0 in x0s:
        soln = solve_ivp(partial(toggleswitch,
                                 alpha=alpha,
                                 beta=beta,
                                 gamma=gamma),
                         tspan,x0,t_eval=t_eval)
        line, = axis.plot(soln.y[0,:],soln.y[1,:],linewidth=1)
        axis.scatter(soln.y[0,0],soln.y[1,0],s=9,color = line.get_color())
        

    
x0s = [[1,2],[2,1]]
plot_nullclines_and_trajectories(a[2],x0s,alpha,beta,gamma)

#------------------------------------------------------------------------------
# Exercise 4: Plotting many trajectories
#------------------------------------------------------------------------------

x0s = np.random.uniform(0,3,[50,2])
plot_nullclines_and_trajectories(a[3],x0s,alpha,beta,gamma)

#------------------------------------------------------------------------------
# Exercise 5: Using different parameters
#------------------------------------------------------------------------------

alpha = 0.4
beta = 1
plot_nullclines_and_trajectories(a[4],x0s,alpha,beta,gamma)

alpha = 0.8
beta = 4
plot_nullclines_and_trajectories(a[5],x0s,alpha,beta,gamma)

f.tight_layout()
f.savefig(os.path.join(result_dir,'training4.pdf'))