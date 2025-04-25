from src.study import run_study
import numpy as np
import matplotlib.pyplot as plt

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

tol = 1e-4

nxy = 5
ns = 5
s, u = run_study(nxy, nxy, T, ns, alpha=a, make_plot=False, initial_condition=initial_condition)
s_pnew, u_pnew = s, u
s_tnew, u_tnew = s, u
lin_error = []


# Position indexing
for i in range(50):
    # Range indexing
    print(f'Position Cells: {nxy + 5}')
    for j in range(50):
        print(f'Time steps: {ns + 5}')
        s_tnew, u_tnew = run_study(nxy, nxy, T, ns + 5, alpha=a, make_plot=False, initial_condition=initial_condition, C1=np.array([-2, -2]), C2=np.array([2, 2]))
        lin_error += [np.abs(u_pnew - u_tnew).max()]
        if np.abs(u_pnew - u_tnew).max() <= tol:
            break
        else:
            s_pnew = s_tnew
            u_pnew = u_tnew
            ns += 5
    s_pnew, u_pnew = run_study(nxy + 5, nxy + 5, T, ns, alpha=a, make_plot=False, initial_condition=initial_condition, C1=np.array([-2, -2]), C2=np.array([2, 2]))
    lin_error += [np.abs(u_pnew.max() - u.max()) ]
    if np.abs(u_pnew.max() - u.max()) <= tol:
        s, u = s_pnew, u_pnew
        break
    else:
        s, u = s_pnew, u_pnew
        nxy += 5

s_lin, u_lin = run_study(nxy, nxy, T, ns, alpha=a, make_plot=False, initial_condition=initial_condition, gif_name='linear_refined.gif', C1=np.array([-2, -2]), C2=np.array([2, 2]))

# This analysis so far has been using linear elements. We can switch to quadratic elements and see what happens:

nxy = 5
ns = 5
s, u = run_study(nxy, nxy, T, ns, alpha=a, make_plot=False, initial_condition=initial_condition, degree=2, C1=np.array([-2, -2]), C2=np.array([2, 2]))
s_pnew, u_pnew = s, u
s_tnew, u_tnew = s, u
quad_error = []

# Position indexing
for i in range(50):
    # Range indexing
    print(f'Position Cells: {nxy + 5}')
    for j in range(50):
        print(f'Time steps: {ns + 5}')
        s_tnew, u_tnew = run_study(nxy, nxy, T, ns + 5, alpha=a, make_plot=False, initial_condition=initial_condition, degree=2, C1=np.array([-2, -2]), C2=np.array([2, 2]))
        quad_error += [np.abs(u_pnew - u_tnew).max()]
        if np.abs(u_pnew - u_tnew).max() <= tol:
            break
        else:
            s_pnew = s_tnew
            u_pnew = u_tnew
            ns += 5
    s_pnew, u_pnew = run_study(nxy + 5, nxy + 5, T, ns, alpha=a, make_plot=False, initial_condition=initial_condition, degree=2, C1=np.array([-2, -2]), C2=np.array([2, 2]))
    quad_error += [np.abs(u_pnew.max() - u.max())]
    if np.abs(u_pnew.max() - u.max()) <= tol:
        s, u = s_pnew, u_pnew
        break
    else:
        s, u = s_pnew, u_pnew
        nxy += 5

s_quad, u_quad = run_study(nxy, nxy, T, ns, alpha=a, make_plot=False, initial_condition=initial_condition, degree=2, gif_name='quadratic_refined.gif', C1=np.array([-2, -2]), C2=np.array([2, 2]))
plt.plot(lin_error, label='Linear Elements')
plt.plot(quad_error, label='Quadratic Elements')
plt.xlabel('Iteration Number')
plt.ylabel('Change in Maximum Value')
plt.title('Mesh Refinement of Thermal Simulation')
plt.legend()
plt.savefig('Convergence.png')
# Based on our print-outs, the quadratic element needed significantly less refinement to converge, especially in space.