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
s, un = run_study(250, 250, T, 250, alpha=a, make_plot=False)

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

# Mesh refinement study

# We have three mesh parameters: nx, ny, and ns (acting as a time mesh). We'll be looking at a different starting value:

def initial_condition(x, A=1, s=1):
    return A * np.exp(- (x[0]**2 + x[1]**2)/s)

a = 1
T = 0.01

# We now want to run studies detailing what happens as we refine our mesh. More specifically, how detailed we need to go.
# Refining time steps seems to be the least computationally expensive. So the general plan for now is to:
#   - Start by refining time until it provides minimal improvement
#   - Refine space and check to see if time adds any improvement
#   - Repeat until convergence

nxy = 10
ns = 10
s, u = run_study(nxy, nxy, T, ns, alpha=a, make_plot=False, initial_condition=initial_condition)
s_pnew, u_pnew = s, u
s_tnew, u_tnew = s, u

# Position indexing
for i in range(50):
    # Range indexing
    print(f'Position Cells: {nxy + 5}')
    for j in range(50):
        print(f'Time steps: {ns + 5}')
        s_tnew, u_tnew = run_study(nxy, nxy, T, ns + 5, alpha=a, make_plot=False, initial_condition=initial_condition)
        if np.abs(u_pnew - u_tnew).max() <= 1e-4:
            break
        else:
            s_pnew = s_tnew
            u_pnew = u_tnew
            ns += 5
    s_pnew, u_pnew = run_study(nxy + 5, nxy + 5, T, ns, alpha=a, make_plot=False, initial_condition=initial_condition)
    if np.abs(u_pnew.max() - u.max()) <= 1e-4:
        s, u = s_pnew, u_pnew
        break
    else:
        s, u = s_pnew, u_pnew
        nxy += 5

# s, u = run_study(nxy, nxy, T, ns, alpha=a, make_plot=True, initial_condition=initial_condition)

# Now we need to find a thing where it fails. There's just one problem: We're unconditionally stable
# 
# This setup uses an implicit scheme which, for the thermal problem, CANNOT be numerically unstable no matter the mesh size or time step.
# That doesn't mean you can't get nonsense however. For example if you have a negative alpha, the system will become divergent:

nxy = 50
ns = 50

a = -1
T = 1
s, u = run_study(nxy, nxy, T, ns, alpha=a, make_plot=True)

# This is more so because the equation is unstable than anything, but it's still interesting to see.
# As

pass