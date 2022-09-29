#------------------------------------------------------------------------------
# Modelling in Biology Python Training Exercise 1
#------------------------------------------------------------------------------

# We must import matplotlib.pyplot to allow us to plot our solutions
from distutils.spawn import find_executable
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
# Use pretty Seaborn-style plots if installed
try:
    import seaborn as sns
    sns.set()
except:
    pass
# Import sympy for symbolic solutions
import sympy as sm

# Create directory for results
result_dir = os.path.join(os.getcwd(),'results')
if not os.path.exists(result_dir):
    os.mkdir(result_dir)


# It is generally a good idea to define constants (or parameters) at the
# beginning of your source file, and refer to them by their names
# afterwards.
k = 0.6
boundaries = [0, 10]
initial_value = 4

# First, we define the function which captures the ODE dynamics
def training1numerical(t,x):
    # Change to "def training1numerical(t,x,k)" if you use the
    # alternative solution where you pass k as a parameter
    k = 0.6
    dxdt = - k * x
    return dxdt


#------------------------------------------------------------------------------
# Exercise 1: Numerical integration
#------------------------------------------------------------------------------

# Our goal is to numerically integrate dx/dt = -k*x.
# We start by writing a function which returns dx/dt for any value of x, 
# and any value of the parameter k. The function is defined as above.
# 
# numerically integrate differential equations of the type dx/dt = f(t,x). 

soln = solve_ivp(training1numerical, 
                 boundaries, 
                 (initial_value,),
                 method='RK45')

# You can also use an alternative solution where you pass k as a parameter 
# in the RK45 function. Then you should also modify the
# training1numerical function definition to accept k as parameter, and utilise 
# functools.partial to set its value before calling RK45:
"""
import functools.partial
training1numerical = partial(training1numerical,k=0.6)
soln = solve_ivp(training1numerical, 
                 boundaries, 
                 (initial_value,),
                 method='RK45')
"""

# Now, T contains the successive integration times, and Y the values at
# these times. We can plot the solution :
f,a = plt.subplots(nrows=3,figsize=(8,6))
f.suptitle('Training Exercise 1')
# We need to squeeze soln.y, as it's a 2-d array by default
a[0].plot(soln.t,np.squeeze(soln.y))
a[0].set_xlabel('Time')
a[0].set_ylabel('x')
a[0].legend(['Numerical solution'])
a[0].set_title('Question 1')


#------------------------------------------------------------------------------
# Exercise 2: Analytical solution
#------------------------------------------------------------------------------

# We use sympy to find the analytical solution here as a demonstration.
# Alternatively, it can be solved on paper, and simply calculated as:
# analytical = initial_value*np.exp(-k*soln.t)

# Define sympy symbols for x and t
T,K = sm.symbols('t,k')
# Define x as a function of t
x = sm.Function('x')(T)
# Define dxdt
dx = x.diff(T)
# Define our symbolic ODE
ode = sm.Eq(dx,-K*x)
sol = sm.dsolve(ode)

# Input initial conditions by
# first, substituting in t=0 into the RHS of our solution
rhs0 = sol.args[1].subs({'t': 0})
# Defining a symbol for our initial value of x
x0 = sm.symbols('x0')
# And constructing an equation
eq_init = sm.Eq(x0, rhs0)

# Now to solve this equation, we isolate the constant C1:
C1 = eq_init.args[1]
# And solve
init_solve= sm.solve(eq_init, C1)
# Now we substitute that back into our solution
final = sol.subs(C1, init_solve[0])

# Loop over the time values to calculate the analytical solution, substituting
# in the initial value, K and T into the symbolic equation.
analytical = []
for tt in soln.t:
    analytical.append(np.asarray(
        final.args[1].subs({x0:initial_value,K:k,T:tt}),
        dtype=np.float64))
analytical = np.stack(analytical)


# And plot it ...
a[1].plot(soln.t, analytical, 'r','LineWidth',2)
a[1].plot(soln.t,np.squeeze(soln.y),'b--','LineWidth',2)
a[1].set_xlabel('Time')
a[1].set_ylabel('x')
a[1].legend(['Analytical solution', 'Numerical solution'])
a[1].set_title('Question 2')

# We can calculate the mean square error easily and print it. Instead of
# using "display", we could also omit the ; at the end, but it triggers a
# warning.
mean_square_error = np.mean((np.squeeze(soln.y)-analytical)**2)
print("Mean squared error: {0:0.10f}".format(mean_square_error))

#------------------------------------------------------------------------------
# Exercise 3: Investigation of Runge-Kutta 4(5) step size
#------------------------------------------------------------------------------

# We calculate timesteps chosen by ode45  by substracting two consecutive
# elements of T.
timesteps = soln.t[1:]- soln.t[:-1]

# ... and plot them as a function of T. Note that "timesteps" has one less
# element than T, hence the T(2:end).
a[2].plot(soln.t[1:],timesteps, 'bx')
a[2].set_xlabel('Time')
a[2].set_ylabel('Timestep')
a[2].set_title('Question 3')
f.tight_layout()
f.savefig(os.path.join(result_dir,'training1.pdf'))
f.savefig(os.path.join(result_dir,'training1.svg'))