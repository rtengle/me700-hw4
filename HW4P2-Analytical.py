from src.study import run_study
import numpy as np

# This is a study on the heat equation taken from a tutorial. I've set up a function to do the study automatically.abs

# Analytic Solution

# The heat equation has very trivial closed-form solutions when the initial profile is a sine wave:
#
# u0 = sin(pi x) sin (pi y) --> u = sin(pi x) sin (pi y) e^(-2 a pi^2 t)
# div u = - pi^2 sin(pi x) sin (pi y) e^(-2 a pi^2 t) - pi^2 sin(pi x) sin (pi y) e^(-2 a pi^2 t)
# a div u = -2 a pi^2 sin(pi x) sin (pi y) e^(-2 a pi^2 t)
# du/dt =  -2 a pi^2 sin(pi x) sin (pi y) e^(-2 a pi^2 t)
#
# Here we run a study to verify it

# Run a study
a = 1
T = 0.01
s, un = run_study(250, 250, T, 250, alpha=a, make_plot=False, gif_name='analytical.gif')

# Exact position and temp data
x = s[:,0]
y = s[:,1]

# Analytic solution:

def u_known(x, y, t, a=1):
    return np.sin(np.pi*x)*np.sin(np.pi*y)*np.exp(-2 * a * np.pi**2 * t)

ua = u_known(x, y, T, a=a)
# Calculate difference
ud = un - ua
imax = np.argmax(np.abs(ud))
print(f'Max Absolute Difference: {ud[imax]}')
print(f'Max Relative Difference: {np.abs(ud[imax]/ua[imax])}')

# We can see that both the max absolute and relative difference between the analytical and numerical are very small.