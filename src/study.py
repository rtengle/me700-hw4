import matplotlib as mpl
import pyvista
import ufl
import numpy as np

from petsc4py import PETSc
from mpi4py import MPI

from dolfinx import fem, mesh, io, plot
from dolfinx.fem.petsc import assemble_vector, assemble_matrix, create_vector, apply_lifting, set_bc

# This is the file where the actual study for HW2P2 is defined. I'll be studying the 2D heat equation for each
# because of how similar it is to my final project. The equation is defined as follows:
# 
# dT/dt = alpha del^2 T
#
# This module will set up functions that run the actual study.abs

def initial_condition(x, a=1):
    return a*np.sin(np.pi*x[0])*np.sin(np.pi*x[1])

def run_study(nx, ny, T, ns, alpha=1, initial_condition=initial_condition, C1=np.array([0, 0]), C2 = np.array([1, 1]), make_plot=True, celltype='tri'):
    # Runs the study for the heat equation dT/dt = a del^2 T.
    # nx : Number of x cells
    # ny : Number of y cells
    # T : Final time
    # ns : Numner of time steps
    # initial_condition : Function for initial condition

    # Define temporal parameters
    t = 0  # Start time
    dt = T / ns  # time step size

    # Define mesh
    cell = {'tri':mesh.CellType.triangle, 'quad':mesh.CellType.quadrilateral}
    domain = mesh.create_rectangle(MPI.COMM_WORLD, [C1, C2],
                                [nx, ny], mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", 1))

    u_n = fem.Function(V)
    u_n.name = "u_n"
    u_n.interpolate(initial_condition)

    # Create boundary condition
    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.full(x.shape[1], True, dtype=bool))
    bc = fem.dirichletbc(PETSc.ScalarType(0), fem.locate_dofs_topological(V, fdim, boundary_facets), V)

    # Define solution variable, and interpolate initial solution for visualization in Paraview
    uh = fem.Function(V)
    uh.name = "uh"
    uh.interpolate(initial_condition)

    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    f = fem.Constant(domain, PETSc.ScalarType(0))
    a = u * v * ufl.dx + alpha * dt * ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = (u_n + alpha * dt * f) * v * ufl.dx

    bilinear_form = fem.form(a)
    linear_form = fem.form(L)

    A = assemble_matrix(bilinear_form, bcs=[bc])
    A.assemble()
    b = create_vector(linear_form)

    solver = PETSc.KSP().create(domain.comm)
    solver.setOperators(A)
    solver.setType(PETSc.KSP.Type.PREONLY)
    solver.getPC().setType(PETSc.PC.Type.LU)

    if make_plot:
        pyvista.start_xvfb()

        grid = pyvista.UnstructuredGrid(*plot.vtk_mesh(V))

        plotter = pyvista.Plotter()
        plotter.open_gif("u_time.gif", fps=10)

        grid.point_data["uh"] = uh.x.array
        warped = grid.warp_by_scalar("uh", factor=1)

        viridis = mpl.colormaps.get_cmap("viridis").resampled(25)
        sargs = dict(title_font_size=25, label_font_size=20, fmt="%.2e", color="black",
                    position_x=0.1, position_y=0.8, width=0.8, height=0.1)

        renderer = plotter.add_mesh(warped, show_edges=True, lighting=False,
                                    cmap=viridis, scalar_bar_args=sargs,
                                    clim=[0, max(uh.x.array)])

    for i in range(ns):
        t += dt

        try:
            # Update the right hand side reusing the initial vector
            with b.localForm() as loc_b:
                loc_b.set(0)
            assemble_vector(b, linear_form)

            # Apply Dirichlet boundary condition to the vector
            apply_lifting(b, [bilinear_form], [[bc]])
            b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
            set_bc(b, [bc])

            # Solve linear problem
            solver.solve(b, uh.x.petsc_vec)
            uh.x.scatter_forward()

            # Update solution at previous time step (u_n)
            u_n.x.array[:] = uh.x.array
        except:
            print('Solver failed')
            break

        # Update plot
        if make_plot:
            new_warped = grid.warp_by_scalar("uh", factor=1)
            warped.points[:, :] = new_warped.points
            warped.point_data["uh"][:] = uh.x.array
            plotter.write_frame()
    if make_plot:
        plotter.close()

    return domain.geometry.x, uh.x.array