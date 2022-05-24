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
from dolfinx.mesh import meshtags
from mpi4py import MPI

from dolfinx_contact.meshing import convert_mesh, create_christmas_tree_mesh
from dolfinx_contact.unbiased.nitsche_unbiased import nitsche_unbiased
from dolfinx_contact.cpp import find_candidate_surface_segment

if __name__ == "__main__":
    desc = "Nitsche's method for two elastic bodies using custom assemblers"
    parser = argparse.ArgumentParser(description=desc,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--theta", default=1, type=np.float64, dest="theta",
                        help="Theta parameter for Nitsche, 1 symmetric, -1 skew symmetric, 0 Penalty-like",
                        choices=[1, -1, 0])
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
    parser.add_argument("--load_steps", default=1, type=np.int32, dest="nload_steps",
                        help="Number of steps for gradual loading")
    parser.add_argument("--res", default=0.2, type=np.float64, dest="res",
                        help="Mesh resolution")
    parser.add_argument("--outfile", type=str, default=None, required=False,
                        help="File for appending results", dest="outfile")
    parser.add_argument("--split", type=np.int32, default=1, required=False,
                        help="number of surface segments", dest="split")
    # Parse input arguments or set to defualt values
    args = parser.parse_args()

    # Current formulation uses bilateral contact
    nitsche_parameters = {"gamma": args.gamma, "theta": args.theta}
    physical_parameters = {"E": args.E, "nu": args.nu, "strain": args.plane_strain}
    threed = args.threed
    nload_steps = args.nload_steps
    simplex = args.simplex

    if threed:
        raise RuntimeError("Not yet implemented")
    else:
        split = args.split
        fname = "xmas_tree"
        create_christmas_tree_mesh(filename=fname, res=args.res, split=split)
        convert_mesh(fname, fname, "triangle")
        convert_mesh(f"{fname}", f"{fname}_facets", "line")
        with XDMFFile(MPI.COMM_WORLD, f"{fname}.xdmf", "r") as xdmf:
            mesh = xdmf.read_mesh(name="Grid")
            domain_marker = xdmf.read_meshtags(mesh, name="Grid")
        tdim = mesh.topology.dim
        gdim = mesh.geometry.dim
        mesh.topology.create_connectivity(tdim - 1, 0)
        mesh.topology.create_connectivity(tdim - 1, tdim)
        with XDMFFile(MPI.COMM_WORLD, f"{fname}_facets.xdmf", "r") as xdmf:
            facet_marker = xdmf.read_meshtags(mesh, name="Grid")

        # create meshtags for candidate segments
        mts = [facet_marker]
        cand_facets_0 = np.sort(np.hstack([facet_marker.indices[facet_marker.values == 5 + i] for i in range(split)]))
        cand_facets_1 = np.sort(
            np.hstack([facet_marker.indices[facet_marker.values == 5 + split + i] for i in range(split)]))

        for i in range(split):
            fcts = np.array(find_candidate_surface_segment(
                mesh, facet_marker.indices[facet_marker.values == 5 + split + i], cand_facets_0, 0.8), dtype=np.int32)
            vls = np.full(len(fcts), 5 + 2 * split + i, dtype=np.int32)
            mts.append(meshtags(mesh, tdim - 1, fcts, vls))

        for i in range(split):
            fcts = np.array(find_candidate_surface_segment(
                mesh, facet_marker.indices[facet_marker.values == 5 + i], cand_facets_1, 0.8), dtype=np.int32)
            vls = np.full(len(fcts), 5 + 3 * split + i, dtype=np.int32)
            mts.append(meshtags(mesh, tdim - 1, fcts, vls))

        # zero dirichlet boundary condition on mesh boundary with tag 5
        dirichlet = [(4, lambda x: np.zeros((gdim, x.shape[1])))]

        # traction (neumann) boundary condition on mesh boundary with tag 3
        t = [0.2, 0.3, 0.0]

        def neumann_func(x):
            values = np.zeros((gdim, x.shape[1]))
            for i in range(x.shape[1]):
                values[:, i] = t[:gdim]
            return values
        neumann = [(3, neumann_func)]

        # body forces
        f = [1.0, -0.5, 0.0]

        def force_func(x):
            values = np.zeros((gdim, x.shape[1]))
            for i in range(x.shape[1]):
                values[:, i] = f[:gdim]
            return values

        body_forces = [(1, force_func)]
        # contact surfaces with tags 5, 6
        data = np.arange(5, 5 + 4 * split, dtype=np.int32)
        offsets = np.concatenate([np.array([0, 2 * split], dtype=np.int32),
                                  np.arange(2 * split + 1, 4 * split + 1, dtype=np.int32)])
        surfaces = create_adjacencylist(data, offsets)

        contact_pairs = []
        for i in range(split):
            contact_pairs.append((i, 3 * split + i))
            contact_pairs.append((split + i, 2 * split + i))

        # create initial guess
        def _u_initial(x):
            values = np.zeros((gdim, x.shape[1]))
            for i in range(x.shape[1]):
                values[0, i] = 1.0
            return values
        V = _fem.VectorFunctionSpace(mesh, ("CG", 1))
        u_initial = _fem.Function(V)
        u_initial.interpolate(_u_initial)

    # Solver options
    ksp_tol = 1e-10
    newton_tol = 1e-7
    newton_options = {"relaxation_parameter": 1,
                      "atol": newton_tol,
                      "rtol": newton_tol,
                      "convergence_criterion": "residual",
                      "max_it": 50,
                      "error_on_nonconvergence": True}
    # petsc_options = {"ksp_type": "preonly", "pc_type": "lu"}
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

    solver_outfile = args.outfile if args.ksp else None

    # Apply forces over multiple steps
    log.set_log_level(log.LogLevel.OFF)
    num_newton_its = np.zeros(nload_steps, dtype=int)
    num_krylov_its = np.zeros(nload_steps, dtype=int)
    newton_time = np.zeros(nload_steps, dtype=np.float64)

    for j in range(nload_steps):
        neumann_incr = []
        for bc in neumann:
            increment = (bc[0], lambda x: (j + 1) * bc[1](x) / nload_steps)
            neumann_incr.append(increment)

        body_force_incr = []
        for bf in body_forces:
            increment = (bf[0], lambda x: (j + 1) * bf[1](x) / nload_steps)
            body_force_incr.append(increment)
        # Solve contact problem using Nitsche's method
        with Timer("~Contact: - all"):
            u1, n, krylov_iterations, solver_time = nitsche_unbiased(
                mesh=mesh, mesh_tags=mts, domain_marker=domain_marker,
                surfaces=surfaces, dirichlet=dirichlet, neumann=neumann_incr,
                contact_pairs=contact_pairs, physical_parameters=physical_parameters,
                body_forces=body_force_incr, nitsche_parameters=nitsche_parameters,
                quadrature_degree=args.q_degree, petsc_options=petsc_options,
                newton_options=newton_options, initGuess=u_initial, outfile=solver_outfile)
        u_initial.x.array[:] = u1.x.array[:]
        num_newton_its[j] = n
        num_krylov_its[j] = krylov_iterations
        newton_time[j] = solver_time
        with XDMFFile(mesh.comm, f"results/xmas_{j}.xdmf", "w") as xdmf:
            xdmf.write_mesh(mesh)
            u1.name = "u"
            xdmf.write_function(u1)

    # Write solution to file
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
    print(f"Newton time: {newton_time}", file=outfile)
    print(f"Newton iterations {num_newton_its}, {sum(num_newton_its)}", file=outfile)
    print(f"Krylov iterations {num_krylov_its}, {sum(num_krylov_its)}", file=outfile)
    print("-" * 25, file=outfile)

    if args.outfile is not None:
        outfile.close()
