# Copyright (C) 2022 Sarah Roggendorf
#
# SPDX-License-Identifier:    MIT

import argparse
import sys

import numpy as np
from dolfinx import log
import dolfinx.fem as _fem
from dolfinx.common import TimingType, list_timings, timing, Timer
from dolfinx.graph import create_adjacencylist
from dolfinx.io import XDMFFile
from dolfinx.mesh import meshtags, locate_entities_boundary
from mpi4py import MPI
from petsc4py import PETSc as _PETSc
import ufl

from dolfinx_contact.meshing import convert_mesh, create_christmas_tree_mesh, create_christmas_tree_mesh_3D
from dolfinx_contact.unbiased.nitsche_unbiased import nitsche_unbiased
from dolfinx_contact.helpers import lame_parameters, sigma_func, weak_dirichlet, epsilon
from dolfinx_contact.cpp import find_candidate_surface_segment

if __name__ == "__main__":
    desc = "Nitsche's method for two elastic bodies using custom assemblers"
    parser = argparse.ArgumentParser(description=desc,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--theta", default=1., type=float, dest="theta",
                        help="Theta parameter for Nitsche, 1 symmetric, -1 skew symmetric, 0 Penalty-like",
                        choices=[1., -1., 0.])
    parser.add_argument("--gamma", default=10, type=np.float64, dest="gamma",
                        help="Coercivity/Stabilization parameter for Nitsche condition")
    parser.add_argument("--quadrature", default=5, type=int, dest="q_degree",
                        help="Quadrature degree used for contact integrals")
    _3D = parser.add_mutually_exclusive_group(required=False)
    _3D.add_argument('--3D', dest='threed', action='store_true',
                     help="Use 3D mesh", default=False)
    _timing = parser.add_mutually_exclusive_group(required=False)
    _timing.add_argument('--timing', dest='timing', action='store_true',
                         help="List timings", default=False)
    _ksp = parser.add_mutually_exclusive_group(required=False)
    _ksp.add_argument('--ksp-view', dest='ksp', action='store_true',
                      help="List ksp options", default=False)
    parser.add_argument("--E", default=1e3, type=np.float64, dest="E",
                        help="Youngs modulus of material")
    parser.add_argument("--nu", default=0.1, type=np.float64, dest="nu", help="Poisson's ratio")
    parser.add_argument("--res", default=0.2, type=np.float64, dest="res",
                        help="Mesh resolution")
    parser.add_argument("--outfile", type=str, default=None, required=False,
                        help="File for appending results", dest="outfile")
    parser.add_argument("--split", type=np.int32, default=1, required=False,
                        help="number of surface segments", dest="split")
    # Parse input arguments or set to defualt values
    args = parser.parse_args()

    threed = args.threed
    split = args.split
    mesh_dir = "meshes"
    fname = f"{mesh_dir}/xmas_tree"
    if threed:
        create_christmas_tree_mesh_3D(filename=fname, res=args.res, split=split, n1=81, n2=41)
        convert_mesh(fname, fname, gdim=3)
        with XDMFFile(MPI.COMM_WORLD, f"{fname}.xdmf", "r") as xdmf:
            mesh = xdmf.read_mesh()
            domain_marker = xdmf.read_meshtags(mesh, "cell_marker")
            tdim = mesh.topology.dim
            mesh.topology.create_connectivity(tdim - 1, tdim)
            facet_marker = xdmf.read_meshtags(mesh, "facet_marker")

        marker_offset = 6
        V = _fem.VectorFunctionSpace(mesh, ("CG", 1))
        # Apply zero Dirichlet boundary conditions in z-direction on part of the xmas-tree

        # Find facets for z-Dirichlet bc
        def identifier(x, z):
            return np.logical_and(np.logical_and(np.isclose(x[2], z),
                                                 abs(x[1]) < 0.1), abs(x[0] - 2) < 0.1)
        dirichlet_facets1 = locate_entities_boundary(mesh, tdim - 1, lambda x: identifier(x, 0.0))
        dirichlet_facets2 = locate_entities_boundary(mesh, tdim - 1, lambda x: identifier(x, 1.0))

        # Create Dirichlet bdy conditions
        bdy_dofs_z = _fem.locate_dofs_topological(V.sub(2), tdim - 1, np.hstack([dirichlet_facets1, dirichlet_facets2]))
        bcs = [_fem.dirichletbc(_PETSc.ScalarType(0), bdy_dofs_z, V.sub(2))]
        g = _fem.Constant(mesh, _PETSc.ScalarType((0, 0, 0)))      # zero dirichlet
        t = _fem.Constant(mesh, _PETSc.ScalarType((0.2, 0.5, 0)))  # traction
        f = _fem.Constant(mesh, _PETSc.ScalarType((1.0, 0.5, 0)))  # body force

    else:
        create_christmas_tree_mesh(filename=fname, res=args.res, split=split)
        convert_mesh(fname, fname, gdim=2)
        with XDMFFile(MPI.COMM_WORLD, f"{fname}.xdmf", "r") as xdmf:
            mesh = xdmf.read_mesh()
            tdim = mesh.topology.dim
            domain_marker = xdmf.read_meshtags(mesh, name="cell_marker")
            mesh.topology.create_connectivity(tdim - 1, tdim)
            facet_marker = xdmf.read_meshtags(mesh, name="facet_marker")

        marker_offset = 5
        V = _fem.VectorFunctionSpace(mesh, ("CG", 1))
        bcs = []
        g = _fem.Constant(mesh, _PETSc.ScalarType((0, 0)))     # zero Dirichlet
        t = _fem.Constant(mesh, _PETSc.ScalarType((0.2, 0.5)))  # traction
        f = _fem.Constant(mesh, _PETSc.ScalarType((1.0, 0.5)))  # body force

    gdim = mesh.geometry.dim
    # create meshtags for candidate segments
    mts = [domain_marker, facet_marker]
    cand_facets_0 = np.sort(
        np.hstack([facet_marker.find(marker_offset + i) for i in range(split)]))
    cand_facets_1 = np.sort(
        np.hstack([facet_marker.find(marker_offset + split + i) for i in range(split)]))

    for i in range(split):
        fcts = np.array(find_candidate_surface_segment(
            mesh, facet_marker.find(marker_offset + split + i), cand_facets_0, 0.8), dtype=np.int32)
        vls = np.full(len(fcts), marker_offset + 2 * split + i, dtype=np.int32)
        mts.append(meshtags(mesh, tdim - 1, fcts, vls))

    for i in range(split):
        fcts = np.array(find_candidate_surface_segment(
            mesh, facet_marker.find(marker_offset + i), cand_facets_1, 0.8), dtype=np.int32)
        vls = np.full(len(fcts), marker_offset + 3 * split + i, dtype=np.int32)
        mts.append(meshtags(mesh, tdim - 1, fcts, vls))

    # contact surfaces with tags from marker_offset to marker_offset + 4 * split (split = #segments)
    data = np.arange(marker_offset, marker_offset + 4 * split, dtype=np.int32)
    offsets = np.concatenate([np.array([0, 2 * split], dtype=np.int32),
                              np.arange(2 * split + 1, 4 * split + 1, dtype=np.int32)])
    surfaces = create_adjacencylist(data, offsets)
    # zero dirichlet boundary condition on mesh boundary with tag 5

    # Function, TestFunction, TrialFunction and measures
    u = _fem.Function(V)
    v = ufl.TestFunction(V)
    dx = ufl.Measure("dx", domain=mesh, subdomain_data=domain_marker)
    ds = ufl.Measure("ds", domain=mesh, subdomain_data=facet_marker)

    # Compute lame parameters
    E = args.E
    nu = args.nu
    mu_func, lambda_func = lame_parameters(False)
    mu = mu_func(E, nu)
    lmbda = lambda_func(E, nu)
    sigma = sigma_func(mu, lmbda)

    # Create variational form without contact contributions
    F = ufl.inner(sigma(u), epsilon(v)) * dx

    # Apply weak Dirichlet boundary conditions using Nitsche's method
    gamma = args.gamma
    theta = args.theta
    F = weak_dirichlet(F, u, g, sigma, E * gamma, theta, ds(4))

    # traction (neumann) boundary condition on mesh boundary with tag 3
    F -= ufl.inner(t, v) * ds(3)

    # body forces
    F -= ufl.inner(f, v) * dx(1)

    contact_pairs = []
    for i in range(split):
        contact_pairs.append((i, 3 * split + i))
        contact_pairs.append((split + i, 2 * split + i))

    # create initial guess
    tree_cells = domain_marker.find(1)

    def _u_initial(x):
        values = np.zeros((gdim, x.shape[1]))
        for i in range(x.shape[1]):
            values[0, i] = 0.1
        return values

    u.interpolate(_u_initial, tree_cells)

    # Solver options
    ksp_tol = 1e-10
    newton_tol = 1e-7
    newton_options = {"relaxation_parameter": 1.0,
                      "atol": newton_tol,
                      "rtol": newton_tol,
                      "convergence_criterion": "residual",
                      "max_it": 50,
                      "error_on_nonconvergence": False}

    # In order to use an LU solver for debugging purposes on small scale problems
    # use the following PETSc options: {"ksp_type": "preonly", "pc_type": "lu"}
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
        "pc_gamg_sym_graph": True,
        "pc_gamg_threshold": 1e-3,
        "pc_gamg_square_graph": 2,
    }

    # Solve contact problem using Nitsche's method
    problem_parameters = {"gamma": E * gamma, "theta": theta, "mu": mu, "lambda": lmbda}
    solver_outfile = args.outfile if args.ksp else None
    log.set_log_level(log.LogLevel.OFF)
    with Timer("~Contact: - all"):
        u1, num_its, krylov_iterations, solver_time = nitsche_unbiased(
            ufl_form=F, u=u, markers=mts, contact_data=(surfaces, contact_pairs),
            bcs=bcs, problem_parameters=problem_parameters, newton_options=newton_options,
            petsc_options=petsc_options, outfile=solver_outfile)

    # write solution to file
    with XDMFFile(mesh.comm, "results/xmas.xdmf", "w") as xdmf:
        xdmf.write_mesh(mesh)
        u1.name = "u"
        xdmf.write_function(u1)

    if args.timing:
        list_timings(mesh.comm, [TimingType.wall])

    if args.outfile is None:
        outfile = sys.stdout
    else:
        outfile = open(args.outfile, "a")
    print("-" * 25, file=outfile)
    print(f"Newton options {newton_options}", file=outfile)
    print(f"num_dofs: {u1.function_space.dofmap.index_map_bs*u1.function_space.dofmap.index_map.size_global}"
          + f", {mesh.topology.cell_type}", file=outfile)
    print(f"Newton solver {timing('~Contact: Newton (Newton solver)')[1]}", file=outfile)
    print(f"Krylov solver {timing('~Contact: Newton (Krylov solver)')[1]}", file=outfile)
    print(f"Newton time: {solver_time}", file=outfile)
    print(f"Newton iterations {num_its}, ", file=outfile)
    print(f"Krylov iterations {krylov_iterations},", file=outfile)
    print("-" * 25, file=outfile)

    if args.outfile is not None:
        outfile.close()
