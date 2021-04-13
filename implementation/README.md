# Contact implementations
This folder contains a collection of Python files for contact simulations with DOLFINx.
Information about input parameters to scripts can be found by running `python3 name_of_file.py --help`
- `create_circle.py`: GMSH Python API script for creating a 2D disk mesh 
- `helpers.py`: Various helpers reused in several other files (should not be executed as a standalone script)
- `nitsche_bc_plane_stress_beam_test.py`: Testing Nitsche Dirichlet boundary conditions for linear elasticity equation using a manufactured solution.
- `nitsche_euler_bernoulli.py`: Verification of Nitsche-Dirichlet boundary conditions for linear elasticity solving the Euler-Bernoulli equiation
- `compare_nitsche_snes.py`: 
- `nitsche_one_way.py`: Contains the Nitsche contact implementation used in `compare_nitsche_snes.py`.