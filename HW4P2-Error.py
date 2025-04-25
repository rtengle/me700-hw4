from src.study import run_study
import numpy as np

# Now we need to find a problem where it fails.
# 
# This setup uses an implicit scheme which, for the thermal problem, CANNOT be numerically unstable no matter the mesh size or time step.
# That doesn't mean you can't get nonsense however. For example if you have a negative alpha, the system will become divergent:

nxy = 50
ns = 50

a = -1
T = 1
s, u = run_study(nxy, nxy, T, ns, alpha=a, make_plot=True, gif_name='divergence.gif')

# This is more so because the equation is unstable than anything, but it's still interesting to see.