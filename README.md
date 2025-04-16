# me700-hw4

To run HW4P1, just use the mamba code used to set-up the default fenicsx environment:

```
module load miniconda
mamba create -n fenicsx-env
mamba activate fenicsx-env
mamba install -c conda-forge fenics-dolfinx mpich pyvista
pip install imageio
pip install gmsh
```