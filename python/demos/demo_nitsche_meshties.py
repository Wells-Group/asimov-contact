# Copyright (C) 2022 Sarah Roggendorf
#
# SPDX-License-Identifier:    MIT

import argparse
import sys

import numpy as np
import ufl
from dolfinx import log
from dolfinx.common import TimingType, list_timings
from dolfinx.fem import dirichletbc, Constant, Function, locate_dofs_topological, VectorFunctionSpace
from dolfinx.graph import create_adjacencylist
from dolfinx.io import XDMFFile
from dolfinx.mesh import locate_entities_boundary, meshtags
from mpi4py import MPI
from petsc4py.PETSc import ScalarType

from dolfinx_contact.helpers import lame_parameters, epsilon, sigma_func
from dolfinx_contact.meshing import (convert_mesh,
                                     create_box_mesh_3D)
from dolfinx_contact.meshtie import nitsche_meshtie
from dolfinx_contact.parallel_mesh_ghosting import create_contact_mesh

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
    fname = "box_3D"
    create_box_mesh_3D(f"{fname}.msh", simplex, gap=gap, W=H)
    convert_mesh(fname, fname, gdim=3)

    with XDMFFile(MPI.COMM_WORLD, f"{fname}.xdmf", "r") as xdmf:
        mesh = xdmf.read_mesh()

    tdim = mesh.topology.dim
    gdim = mesh.geometry.dim
    mesh.topology.create_connectivity(tdim - 1, 0)
    mesh.topology.create_connectivity(tdim - 1, tdim)

    neumann_bdy = 1
    contact_bdy_1 = 2
    contact_bdy_2 = 3
    dirichlet_bdy = 4
    # Create meshtag for top and bottom markers
    top_facets1 = locate_entities_boundary(mesh, tdim - 1, lambda x: np.isclose(x[2], H, atol=1e-10))
    bottom_facets1 = locate_entities_boundary(
        mesh, tdim - 1, lambda x: np.isclose(x[2], 0.0, atol=1e-10))
    top_facets2 = locate_entities_boundary(mesh, tdim - 1, lambda x: np.isclose(x[2], -gap, atol=1e-10))
    bottom_facets2 = locate_entities_boundary(
        mesh, tdim - 1, lambda x: np.isclose(x[2], -H - gap, atol=1e-10))
    top_values = np.full(len(top_facets1), neumann_bdy, dtype=np.int32)
    bottom_values = np.full(
        len(bottom_facets1), contact_bdy_1, dtype=np.int32)

    surface_values = np.full(len(top_facets2), contact_bdy_2, dtype=np.int32)
    sbottom_values = np.full(
        len(bottom_facets2), dirichlet_bdy, dtype=np.int32)
    indices = np.concatenate([top_facets1, bottom_facets1, top_facets2, bottom_facets2])
    values = np.hstack([top_values, bottom_values, surface_values, sbottom_values])
    sorted_facets = np.argsort(indices)
    facet_marker = meshtags(mesh, tdim - 1, indices[sorted_facets], values[sorted_facets])

    # mark the whole domain
    cells = np.arange(mesh.topology.index_map(tdim).size_local
                      + mesh.topology.index_map(tdim).num_ghosts, dtype=np.int64)
    domain_marker = meshtags(mesh, tdim, cells, np.full(cells.shape, 1, dtype=np.int32))

    if mesh.comm.size > 1:
        mesh, facet_marker, domain_marker = create_contact_mesh(
            mesh, facet_marker, domain_marker, [contact_bdy_1, contact_bdy_2])

    # Function, TestFunction, TrialFunction and measures
    V = VectorFunctionSpace(mesh, ("CG", 1))
    u = Function(V)
    v = ufl.TestFunction(V)
    w = ufl.TrialFunction(V)
    dx = ufl.Measure("dx", domain=mesh, subdomain_data=domain_marker)
    ds = ufl.Measure("ds", domain=mesh, subdomain_data=facet_marker)
    h = ufl.CellDiameter(mesh)
    n = ufl.FacetNormal(mesh)

    # Compute lame parameters
    E = args.E
    nu = args.nu
    mu_func, lambda_func = lame_parameters(False)
    mu = mu_func(E, nu)
    lmbda = lambda_func(E, nu)
    sigma = sigma_func(mu, lmbda)

    # dictionary with problem parameters
    gamma = args.gamma
    theta = args.theta
    problem_parameters = {"mu": mu, "lambda": lmbda, "gamma": E * gamma, "theta": theta}

    J = ufl.inner(sigma(w), epsilon(v)) * dx

    # traction (neumann) boundary condition on mesh boundary with tag 3
    t = Constant(mesh, ScalarType((0.0, 0.5, 0.0)))
    F = ufl.inner(t, v) * ds(neumann_bdy)

    # Dirichlet bdry conditions
    g = Constant(mesh, ScalarType((0.0, 0.0, 0.0)))
    if args.lifting:
        bdy_dofs = locate_dofs_topological(V, tdim - 1, facet_marker.find(dirichlet_bdy))  # type: ignore
        bcs = [dirichletbc(g, bdy_dofs, V)]
    else:
        bcs = []
        J += - ufl.inner(sigma(w) * n, v) * ds(dirichlet_bdy)\
            - theta * ufl.inner(sigma(v) * n, w) * \
            ds(dirichlet_bdy) + E * gamma / h * ufl.inner(w, v) * ds(dirichlet_bdy)
        F += - theta * ufl.inner(sigma(v) * n, g) * \
            ds(dirichlet_bdy) + E * gamma / h * ufl.inner(g, v) * ds(dirichlet_bdy)

    # body forces
    f = Constant(mesh, ScalarType((0.0, 0.5, 0.0)))
    F += ufl.inner(f, v) * dx

    # Solver options
    ksp_tol = 1e-10

    # for debugging use petsc_options = {"ksp_type": "preonly", "pc_type": "lu"}
    petsc_options = {
        "matptap_via": "scalable",
        "ksp_type": "cg",
        "ksp_rtol": ksp_tol,
        "ksp_atol": ksp_tol,
        "pc_type": "gamg",
        "pc_mg_levels": 3,
        "mg_levels_ksp_type": "chebyshev",
        "mg_levels_pc_type": "jacobi",
        "pc_gamg_type": "agg",
        "pc_gamg_coarse_eq_limit": 100,
        "pc_gamg_agg_nsmooths": 1,
        "pc_gamg_threshold": 1e-3,
        "pc_gamg_square_graph": 2,
    }
    # Pack mesh data for Nitsche solver
    contact = [(1, 0), (0, 1)]
    data = np.array([contact_bdy_1, contact_bdy_2], dtype=np.int32)
    offsets = np.array([0, 2], dtype=np.int32)
    surfaces = create_adjacencylist(data, offsets)

    log.set_log_level(log.LogLevel.OFF)
    solver_outfile = args.outfile if args.ksp else None

    # Solve contact problem using Nitsche's method
    u, krylov_iterations, solver_time, _ = nitsche_meshtie(lhs=J, rhs=F, u=u, markers=[domain_marker, facet_marker],
                                                           surface_data=(surfaces, contact),
                                                           bcs=bcs, problem_parameters=problem_parameters,
                                                           petsc_options=petsc_options)

    # Reset mesh to initial state and write accumulated solution
    with XDMFFile(mesh.comm, "results/u_meshtie.xdmf", "w") as xdmf:
        xdmf.write_mesh(mesh)
        u.name = "u"
        xdmf.write_function(u)
    if args.timing:
        list_timings(mesh.comm, [TimingType.wall])

    if args.outfile is None:
        outfile = sys.stdout
    else:
        outfile = open(args.outfile, "a")
    print("-" * 25, file=outfile)
    print(f"num_dofs: {u.function_space.dofmap.index_map_bs*u.function_space.dofmap.index_map.size_global}"
          + f", {mesh.topology.cell_type}", file=outfile)
    print(f"Krylov solver {solver_time}", file=outfile)
    print(f"Krylov iterations {krylov_iterations}", file=outfile)
    print("-" * 25, file=outfile)

    if args.outfile is not None:
        outfile.close()
