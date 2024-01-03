# Copyright (C) 2022 Sarah Roggendorf
#
# SPDX-License-Identifier:    MIT

import argparse
import sys

import numpy as np
import ufl
from dolfinx import log, default_scalar_type
from dolfinx.common import TimingType, list_timings, Timer, timing
from dolfinx.fem import (dirichletbc, Constant, form, Function, FunctionSpace,
                         locate_dofs_topological)
from dolfinx.fem.petsc import (apply_lifting, assemble_vector, assemble_matrix,
                               create_vector, set_bc)
from dolfinx.graph import adjacencylist
from dolfinx.io import XDMFFile
from mpi4py import MPI
from petsc4py import PETSc

from dolfinx_contact.helpers import near_nullspace_subdomains
from dolfinx_contact.meshing import (convert_mesh,
                                     create_box_mesh_3D)
from dolfinx_contact.parallel_mesh_ghosting import create_contact_mesh
from dolfinx_contact.cpp import MeshTie, Problem

if __name__ == "__main__":
    desc = "Nitsche's method for two elastic bodies using custom assemblers"
    parser = argparse.ArgumentParser(description=desc,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--theta", default=1., type=float, dest="theta",
                        help="Theta parameter for Nitsche, 1 symmetric, -1 skew symmetric, 0 Penalty-like",
                        choices=[1., -1., 0.])
    parser.add_argument("--gamma", default=10, type=float, dest="gamma",
                        help="Coercivity/Stabilization parameter for Nitsche condition")
    parser.add_argument("--quadrature", default=5, type=int, dest="q_degree",
                        help="Quadrature degree used for contact integrals")
    _timing = parser.add_mutually_exclusive_group(required=False)
    _timing.add_argument('--timing', dest='timing', action='store_true',
                         help="List timings", default=False)
    _ksp = parser.add_mutually_exclusive_group(required=False)
    _ksp.add_argument('--ksp-view', dest='ksp', action='store_true',
                      help="List ksp options", default=False)
    _simplex = parser.add_mutually_exclusive_group(required=False)
    _simplex.add_argument('--simplex', dest='simplex', action='store_true',
                          help="Use triangle/tet mesh", default=False)
    parser.add_argument("--E", default=1e3, type=np.float64, dest="E",
                        help="Youngs modulus of material")
    parser.add_argument("--nu", default=0.1, type=np.float64, dest="nu", help="Poisson's ratio")
    parser.add_argument("--outfile", type=str, default=None, required=False,
                        help="File for appending results", dest="outfile")
    _lifting = parser.add_mutually_exclusive_group(required=False)
    _lifting.add_argument('--lifting', dest='lifting', action='store_true',
                          help="Apply lifting (strong enforcement of Dirichlet condition",
                          default=False)

    # Parse input arguments or set to defualt values
    args = parser.parse_args()
    simplex = args.simplex

    # Load mesh and create identifier functions for the top (Displacement condition)
    # and the bottom (contact condition)
    displacement = [[0, 0, 0]]
    gap = 1e-5
    H = 1.5
    fname = "meshes/box_3D"
    create_box_mesh_3D(f"{fname}.msh", simplex, gap=gap, W=H, offset=0.0)
    convert_mesh(fname, fname, gdim=3)

    with XDMFFile(MPI.COMM_WORLD, f"{fname}.xdmf", "r") as xdmf:
        mesh = xdmf.read_mesh()
        domain_marker = xdmf.read_meshtags(mesh, "cell_marker")
        tdim = mesh.topology.dim
        mesh.topology.create_connectivity(tdim - 1, tdim)
        facet_marker = xdmf.read_meshtags(mesh, "facet_marker")

    tdim = mesh.topology.dim
    gdim = mesh.geometry.dim
    mesh.topology.create_connectivity(tdim - 1, 0)
    mesh.topology.create_connectivity(tdim - 1, tdim)

    if simplex:
        neumann_bdy = 7
        contact_bdy_1 = 6
        contact_bdy_2 = 13
        dirichlet_bdy = 12
    else:
        neumann_bdy = 7
        contact_bdy_1 = 2
        contact_bdy_2 = 13
        dirichlet_bdy = 8

    if mesh.comm.size > 1:
        mesh, facet_marker, domain_marker = create_contact_mesh(
            mesh, facet_marker, domain_marker, [contact_bdy_1, contact_bdy_2])

    # Function, TestFunction, TrialFunction and measures
    V = FunctionSpace(mesh, ("Lagrange", 1))
    v = ufl.TestFunction(V)
    w = ufl.TrialFunction(V)
    dx = ufl.Measure("dx", domain=mesh, subdomain_data=domain_marker)
    ds = ufl.Measure("ds", domain=mesh, subdomain_data=facet_marker)
    h = ufl.CellDiameter(mesh)
    n = ufl.FacetNormal(mesh)

    # Nitsche parameters
    gamma = args.gamma
    theta = args.theta

    # bilinear form
    kdt_val = 5
    V0 = FunctionSpace(mesh, ("DG", 0))
    kdt = Function(V0)
    kdt.interpolate(lambda x: np.full((1, x.shape[1]), kdt_val))
    J = kdt * ufl.inner(ufl.grad(w), ufl.grad(v)) * dx

    # source term
    f = Constant(mesh, default_scalar_type(1.0))
    F = ufl.inner(f, v) * dx

    # Neumann (flux) boundary condition on mesh boundary with tag 3
    t = Constant(mesh, default_scalar_type(0.5))
    F += ufl.inner(t, v) * ds(neumann_bdy)

    # Dirichlet bdry conditions
    g = Constant(mesh, default_scalar_type((0.0)))

    bdy_dofs = locate_dofs_topological(V, tdim - 1, facet_marker.find(dirichlet_bdy))  # type: ignore
    bcs = [dirichletbc(g, bdy_dofs, V)]

    # compile forms
    cffi_options = ["-Ofast", "-march=native"]
    jit_options = {"cffi_extra_compile_args": cffi_options, "cffi_libraries": ["m"]}
    F = form(F, jit_options=jit_options)
    J = form(J, jit_options=jit_options)

    # Solver options
    ksp_tol = 1e-10

    # for debugging: petsc_options = {"ksp_type": "preonly", "pc_type": "lu"}
    petsc_options = {
        "matptap_via": "scalable",
        "ksp_type": "cg",
        "ksp_rtol": ksp_tol,
        "ksp_atol": ksp_tol,
        "pc_type": "gamg",
        "pc_mg_levels": 3,
        "pc_mg_cycles": 1,   # 1 is v, 2 is w
        "mg_levels_ksp_type": "chebyshev",
        "mg_levels_pc_type": "jacobi",
        "pc_gamg_type": "agg",
        "pc_gamg_coarse_eq_limit": 100,
        "pc_gamg_agg_nsmooths": 1,
        "pc_gamg_threshold": 1e-3,
        "pc_gamg_square_graph": 2,
        "pc_gamg_reuse_interpolation": False,
        "ksp_norm_type": "unpreconditioned"
    }
    # Pack mesh data for Nitsche solver
    contact = [(1, 0), (0, 1)]
    data = np.array([contact_bdy_1, contact_bdy_2], dtype=np.int32)
    offsets = np.array([0, 2], dtype=np.int32)
    surfaces = adjacencylist(data, offsets)

    log.set_log_level(log.LogLevel.OFF)
    solver_outfile = args.outfile if args.ksp else None

    # initialise meshties
    meshties = MeshTie([facet_marker._cpp_object], surfaces, contact,
                       mesh._cpp_object, quadrature_degree=5)
    meshties.generate_kernel_data(Problem.Poisson, V._cpp_object, {"kdt": kdt._cpp_object}, gamma, theta)

    # create matrix, vector
    A = meshties.create_matrix(J._cpp_object)
    b = create_vector(F)

    # Assemble right hand side
    b.zeroEntries()
    b.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)  # type: ignore
    assemble_vector(b, F)

    # Apply boundary condition and scatter reverse
    if len(bcs) > 0:
        apply_lifting(b, [J], bcs=[bcs], scale=-1.0)
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)  # type: ignore
    if len(bcs) > 0:
        set_bc(b, bcs)

    # Assemble matrix
    A.zeroEntries()
    meshties.assemble_matrix(A, V._cpp_object, Problem.Poisson)
    assemble_matrix(A, J, bcs=bcs)  # type: ignore
    A.assemble()

    # Set piecewise constant nullspace
    null_space = near_nullspace_subdomains(V, domain_marker, np.unique(domain_marker.values),
                                           num_domains=2)
    A.setNearNullSpace(null_space)

    # Create PETSc Krylov solver and turn convergence monitoring on
    opts = PETSc.Options()  # type: ignore
    for key in petsc_options:
        opts[key] = petsc_options[key]
    solver = PETSc.KSP().create(mesh.comm)  # type: ignore
    solver.setFromOptions()

    # Set matrix operator
    solver.setOperators(A)

    uh = Function(V)

    dofs_global = V.dofmap.index_map_bs * V.dofmap.index_map.size_global
    log.set_log_level(log.LogLevel.OFF)
    # Set a monitor, solve linear system, and display the solver
    # configuration
    solver.setMonitor(lambda _, its, rnorm: print(f"Iteration: {its}, rel. residual: {rnorm}"))
    timing_str = "~Contact : Krylov Solver"
    with Timer(timing_str):
        solver.solve(b, uh.vector)

    # Scatter forward the solution vector to update ghost values
    uh.x.scatter_forward()

    solver_time = timing(timing_str)[1]
    print(f"{dofs_global}\n",
          f"Number of Krylov iterations {solver.getIterationNumber()}\n",
          f"Solver time {solver_time}", flush=True)

    # Reset mesh to initial state and write accumulated solution
    with XDMFFile(mesh.comm, "results/poisson.xdmf", "w") as xdmf:
        xdmf.write_mesh(mesh)
        uh.name = "u"
        xdmf.write_function(uh)
    if args.timing:
        list_timings(mesh.comm, [TimingType.wall])

    if args.outfile is None:
        outfile = sys.stdout
    else:
        outfile = open(args.outfile, "a")
    print("-" * 25, file=outfile)
    print(f"num_dofs: {uh.function_space.dofmap.index_map_bs*uh.function_space.dofmap.index_map.size_global}"
          + f", {mesh.topology.cell_types[0]}", file=outfile)
    print(f"Krylov solver {solver_time}", file=outfile)
    print(f"Krylov iterations {solver.getIterationNumber()}", file=outfile)
    print("-" * 25, file=outfile)

    if args.outfile is not None:
        outfile.close()
