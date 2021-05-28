import argparse

import dolfinx
import dolfinx.io
import numpy as np
from mpi4py import MPI

from nitsche_rigid_surface import nitsche_rigid_surface

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare Nitsche's metood for contact against a straight plane"
                                     + " with PETSc SNES",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--theta", default=1, type=np.float64, dest="theta",
                        help="Theta parameter for Nitsche, 1 symmetric, -1 skew symmetric, 0 Penalty-like")
    parser.add_argument("--gamma", default=1000, type=np.float64, dest="gamma",
                        help="Coercivity/Stabilization parameter for Nitsche condition")
    _solve = parser.add_mutually_exclusive_group(required=False)
    _solve.add_argument('--linear', dest='linear_solver', action='store_true',
                        help="Use linear solver", default=False)
    _3D = parser.add_mutually_exclusive_group(required=False)
    _3D.add_argument('--3D', dest='threed', action='store_true',
                     help="Use 3D mesh", default=False)
    _strain = parser.add_mutually_exclusive_group(required=False)
    _strain.add_argument('--strain', dest='plane_strain', action='store_true',
                         help="Use plane strain formulation", default=False)
    _dirichlet = parser.add_mutually_exclusive_group(required=False)
    _dirichlet.add_argument('--dirichlet', dest='dirichlet', action='store_true',
                            help="Use strong Dirichlet formulation", default=False)
    _E = parser.add_argument("--E", default=1e3, type=np.float64, dest="E",
                             help="Youngs modulus of material")
    _nu = parser.add_argument(
        "--nu", default=0.1, type=np.float64, dest="nu", help="Poisson's ratio")
    _disp = parser.add_argument("--disp", default=0.08, type=np.float64, dest="disp",
                                help="Displacement BC in negative y direction")
    _ref = parser.add_argument("--refinements", default=2, type=np.int32,
                               dest="refs", help="Number of mesh refinements")
    _gap = parser.add_argument(
        "--gap", default=0.0, type=np.float64, dest="gap", help="Gap between plane and y=0")
    # Parse input arguments or set to defualt values
    args = parser.parse_args()

    # Current formulation uses unilateral contact, i.e. s is unused
    nitsche_parameters = {"gamma": args.gamma, "theta": args.theta, "s": 0}
    nitsche_bc = not args.dirichlet
    physical_parameters = {"E": args.E, "nu": args.nu, "strain": args.plane_strain}
    vertical_displacement = -args.disp
    num_refs = args.refs + 1
    gap = args.gap
    top_value = 1
    threed = args.threed
    bottom_value = 2

    # Load mesh and create identifier functions for the top (Displacement condition)
    # and the bottom (contact condition)
    fname = "twomeshes"
    with dolfinx.io.XDMFFile(MPI.COMM_WORLD, f"{fname}.xdmf", "r") as xdmf:
        mesh = xdmf.read_mesh(name="Grid")
    with dolfinx.io.XDMFFile(MPI.COMM_WORLD, "test.xdmf", "w") as xdmf:
        xdmf.write_mesh(mesh)
    tdim = mesh.topology.dim
    mesh.topology.create_connectivity(tdim - 1, 0)
    mesh.topology.create_connectivity(tdim - 1, tdim)
    with dolfinx.io.XDMFFile(MPI.COMM_WORLD, f"{fname}_facets.xdmf", "r") as xdmf:
        facet_marker = xdmf.read_meshtags(mesh, name="Grid")

    e_abs = []
    e_rel = []
    dofs_global = []
    rank = MPI.COMM_WORLD.rank
    top_value = 2
    bottom_value = 4
    surface_value = 9
    surface_bottom = 7
    mesh_data = (facet_marker, top_value, bottom_value, surface_value, surface_bottom)
    # Solve contact problem using Nitsche's method
    u1 = nitsche_rigid_surface(mesh=mesh, mesh_data=mesh_data, physical_parameters=physical_parameters,
                               vertical_displacement=vertical_displacement, refinement=0, nitsche_bc=False)
