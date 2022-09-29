#------------------------------------------------------------------------------
# Modelling in Biology Python Training Exercise 3
#------------------------------------------------------------------------------
from distutils.spawn import find_executable
from functools import partial
import matplotlib
import matplotlib.pyplot as plt
# Check if we can use pretty LATEX formatting in plots
if find_executable('latex') and matplotlib.checkdep_usetex(True):
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
# We use multiprocessing to speed up the calculation of (many) trajectories
import multiprocessing as mp
import os
import numpy as np
# Use pretty Seaborn-style plots if installed
try:
    import seaborn as sns
    sns.set()
except:
    pass
# Use tqdm for tracking potentially long runtimes
try: 
    from tqdm import tqdm
    has_tqdm=True
except:
    has_tqdm=False

# Create directory for results
result_dir = os.path.join(os.getcwd(),'results')
if not os.path.exists(result_dir):
    os.mkdir(result_dir)


def sdefun(x0,t,s,h,k):
    # x: array containing complete trajectory, 
    # t: array containing corresponding time points.
    # x0: Initial condition, t: time range, s: sigma.

    x = np.zeros(np.shape(t))
    x[0]=x0
    for i in range(1,len(t)):     # Calculate next point
        x[i] = x[i-1]- h*k*x[i-1] + s*np.sqrt(h) * np.random.normal()
    return x


# Variable declarations
k =  3/16 
h = 0.01 
x0 = 6
t = np.arange(0,10,h)
s = 0.2


#------------------------------------------------------------------------------
# Exercise 1a: Plot superimposed trajectories
#------------------------------------------------------------------------------

trajectories = []
f,a = plt.subplots()
for i in range(20):
    x = sdefun(x0,t,s,h,k) # Generate the trajectory.
    a.plot(t,x)             # Plot the trajectory.
    trajectories.append(x) # Store the trajectory.

a.set_xlabel('Time') 
a.set_ylabel('x') 
a.set_title('1a.') # Figure labelling.
f.tight_layout()
f.savefig(os.path.join(result_dir,'training3_exercise1a.pdf'))

#------------------------------------------------------------------------------
# Exercise 1b: Averaging trajectories and calculating error
#------------------------------------------------------------------------------

x_mean = np.mean(trajectories,axis=0)   # Compute the mean trajectory.
x_det = sdefun(x0,t,0,h,k)            # Compute the deterministic trajectory (approximately).
MSE = np.mean((x_mean-x_det)**2) # Compute the mean squared error between the above two.

#------------------------------------------------------------------------------
# Exercise 2a: Long runtime and histogram calculation
#------------------------------------------------------------------------------

# Variable declarations
k =  3/16 
h = 0.01 
x0 = 0
t = np.arange(0,200,h)              # lets say 200
s = 0.1

num_trajectories = 2000

# We define a function to run the simulation and return the final value.
# The function must take a dummy variable, i, passed to it from the mp.Pool()
# instance. In other situations, such as performing a parallel parameter scan,
# this dummy variable can be replaced by the parameter of choice.
def run_and_get_final(i,x0,t,s,h,k):
    x = sdefun(x0,t,s,h,k)
    return x[-1]

# Use functools.partial to set the fixed parameters of the run function
mp_function = partial(run_and_get_final,
                      x0=x0,
                      t=t,
                      s=s,
                      h=h,
                      k=k)

# We use the total number of CPUs available to run the simulations in parallel.
# Your computer might be a bit slow while this runs!
with mp.Pool(os.cpu_count()) as p:
    # We wrap the call to p.imap with tqdm, which allows the progressbar to
    # track progress in a multiprocessing call! Note: this only works for 
    # mp.Pool().imap() and mp.Pool().imap_unordered(), which return values
    # once ready, rather than blocking until finished as other Pool() methods
    # do.
    # We also manually set a chunksize to improve efficiency; see
    # https://docs.python.org/3/library/multiprocessing.html
    finals = list(tqdm(p.imap(mp_function,range(num_trajectories),
                                   chunksize=40),
                            total = num_trajectories,
                            desc="Exercise 2a"))


""" # This is a serial version of the code (i.e. doesn't use multiprocessing)
finals = np.zeros(num_trajectories)
# Safely try to use tqdm
it = range(num_trajectories)
if has_tqdm:
    it = tqdm(it)

for i in it:
    x = sdefun(x0,t,s,h,k)
    finals[i] = x[-1]
"""
    

f,a = plt.subplots()
a.hist(finals,40)   # Generate the histogram -- using 40 bins.
a.set_xlabel('x(T)') 
a.set_ylabel('Frequency') 
a.set_title(r'2a. $\sigma$ = 0.1') # Figure labelling.

mu_2a = np.mean(finals) # Estimate the mean using the sample mean--see, https://en.wikipedia.org/wiki/Sample_mean_and_sample_covariance.
stddev_2a = np.std(finals) # Estimate the standard deviation using the sample standard deviation--https://en.wikipedia.org/wiki/Unbiased_estimation_of_standard_deviation#Background.

a.text(0.1,0.9,r"$\mu \approx$ " f"{mu_2a:.3f}", transform=a.transAxes)
a.text(0.1,0.8,r"$\sigma \approx$ " f"{stddev_2a:.3f}", transform=a.transAxes)
f.tight_layout()
f.savefig(os.path.join(result_dir,'training3_exercise2a.pdf'))

#------------------------------------------------------------------------------
# Exercise 2b: Long runtime, sigma=5
#------------------------------------------------------------------------------

s = 5 # Change sigma to 5

# Use functools.partial to set the fixed parameters of the run function
mp_function = partial(run_and_get_final,
                      x0=x0,
                      t=t,
                      s=s,
                      h=h,
                      k=k)

# We use the total number of CPUs available to run the simulations in parallel.
# Your computer might be a bit slow while this runs!
with mp.Pool(os.cpu_count()) as p:
    # We wrap the call to p.imap with tqdm, which allows the progressbar to
    # track progress in a multiprocessing call! Note: this only works for 
    # mp.Pool().imap() and mp.Pool().imap_unordered(), which return values
    # once ready, rather than blocking until finished as other Pool() methods
    # do.
    # We also manually set a chunksize to improve efficiency; see
    # https://docs.python.org/3/library/multiprocessing.html
    finals = list(tqdm(p.imap(mp_function,range(num_trajectories),
                                   chunksize=40),
                            total = num_trajectories,
                            desc="Exercise 2b"))

""" # This is a serial version of the code (i.e. doesn't use multiprocessing)
finals = np.zeros(num_trajectories)
# Safely try to use tqdm
it = range(num_trajectories)
if has_tqdm:
    it = tqdm(it)
for i in it:
    x = sdefun(x0,t,s,h,k)
    finals[i] = x[-1]
"""

f,a = plt.subplots()
a.hist(finals,40)   # Generate the histogram -- using 40 bins.
a.set_xlabel('x(T)') 
a.set_ylabel('Frequency') 
a.set_title(r'2b. $\sigma$ = 5.') # Figure labelling.

mu_2b = np.mean(finals)
stddev_2b = np.std(finals)

a.text(0.1,0.9,r"$\mu \approx$ " f"{mu_2b:.3f}", transform=a.transAxes)
a.text(0.1,0.8,r"$\sigma \approx$ " f"{stddev_2b:.3f}", transform=a.transAxes)
f.tight_layout()
f.savefig(os.path.join(result_dir,'training3_exercise2b.pdf'))

#------------------------------------------------------------------------------
# Exercise 2c: Discussion of histograms with respect to mean and stddev
#------------------------------------------------------------------------------

print("Sigma = 0.1: Mean = {0:.5f}, Stddev = {1:.5f}".format(mu_2a, stddev_2a))
print("Sigma = 5.: Mean = {0:.5f}, Stddev = {1:.5f}".format(mu_2b, stddev_2b))
