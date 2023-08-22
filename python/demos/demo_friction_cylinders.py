# Copyright (C) 2023 Sarah Roggendorf
#
# SPDX-License-Identifier:    MIT

import argparse

import numpy as np
import ufl
from dolfinx.io import XDMFFile, VTXWriter
from dolfinx.fem import (Constant, dirichletbc, Function, FunctionSpace,
                         VectorFunctionSpace, locate_dofs_topological)
from dolfinx.graph import create_adjacencylist
from dolfinx.mesh import locate_entities
from mpi4py import MPI
from petsc4py.PETSc import ScalarType

from dolfinx_contact.helpers import (epsilon, sigma_func, lame_parameters)
from dolfinx_contact.meshing import (convert_mesh,
                                     create_quarter_disks_mesh)
from dolfinx_contact.cpp import ContactMode


from dolfinx_contact.unbiased.nitsche_unbiased import nitsche_unbiased
from dolfinx_contact.unbiased.contact_problem import create_contact_solver

if __name__ == "__main__":
    desc = "Friction example with two elastic cylinders for verifying correctness of code"
    parser = argparse.ArgumentParser(description=desc,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--quadrature", default=5, type=int, dest="q_degree",
                        help="Quadrature degree used for contact integrals")
    parser.add_argument("--order", default=1, type=int, dest="order",
                        help="Order of mesh geometry", choices=[1, 2])
    _simplex = parser.add_mutually_exclusive_group(required=False)
    _simplex.add_argument('--simplex', dest='simplex', action='store_true',
                          help="Use triangle/tet mesh", default=False)
    parser.add_argument("--res", default=0.1, type=np.float64, dest="res",
                        help="Mesh resolution")

    # Parse input arguments or set to defualt values
    args = parser.parse_args()
    simplex = args.simplex
    mesh_dir = "meshes"

    # Problem parameters
    R = 8
    gap = 0.01
    p = 0.625
    E = 200
    nu = 0.3
    mu_func, lambda_func = lame_parameters(True)
    mu = mu_func(E, nu)
    lmbda = lambda_func(E, nu)

    # Create mesh
    outname = "results/friction_cyl_2D_simplex" if simplex else "results/friction_cyl_2D_quads"
    fname = f"{mesh_dir}/friction_cyl_2D_simplex" if simplex else f"{mesh_dir}/friction_cyl_2D_quads"
    create_quarter_disks_mesh(f"{fname}.msh", args.res, args.order, not simplex, R, gap)

    convert_mesh(fname, f"{fname}.xdmf", gdim=2)
    with XDMFFile(MPI.COMM_WORLD, f"{fname}.xdmf", "r") as xdmf:
        mesh = xdmf.read_mesh()
        domain_marker = xdmf.read_meshtags(mesh, name="cell_marker")
        tdim = mesh.topology.dim
        mesh.topology.create_connectivity(tdim - 1, tdim)
        facet_marker = xdmf.read_meshtags(mesh, name="facet_marker")

    vals = facet_marker.values
    facet_marker.values[vals == 5] = 4
    facet_marker.values[vals == 8] = 7
    facet_marker.values[vals == 11] = 10
    facet_marker.values[vals == 14] = 13

    contact_bdy_1 = 4
    contact_bdy_2 = 10
    top = 7
    bottom = 13

    # theoretical solution
    a = 2 * np.sqrt(2 * R**2 * p * (1 - nu**2) / (np.pi * E))
    p0 = 4 * R * p / (np.pi * a)

    def _pressure(x):
        vals = np.zeros(x.shape[1])
        for i in range(x.shape[1]):
            if abs(x[0][i]) < a:
                vals[i] = p0 * np.sqrt(1 - x[0][i]**2 / a**2)
        return vals

    # Solver options
    ksp_tol = 1e-10
    newton_tol = 1e-7
    newton_options = {"relaxation_parameter": 1.0,
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
        "pc_gamg_threshold": 1e-3,
        "pc_gamg_square_graph": 2,
        "pc_gamg_reuse_interpolation": False,
        "ksp_initial_guess_nonzero": False,
        "ksp_norm_type": "unpreconditioned"
    }

    # Step 1: frictionless contact
    V = VectorFunctionSpace(mesh, ("CG", args.order))
    # boundary conditions
    t = Constant(mesh, ScalarType((0.0, -p)))
    g = Constant(mesh, ScalarType((0.0)))

    symmetry_nodes = locate_entities(mesh, 0, lambda x: np.isclose(x[0], 0))
    dofs_symmetry = locate_dofs_topological(V.sub(0), 0, symmetry_nodes)
    dofs_bottom = locate_dofs_topological(V.sub(1), 1, facet_marker.find(bottom))

    bc_fns = [g, g]
    bcs = [dirichletbc(g, dofs_symmetry, V.sub(0)),
           dirichletbc(g, dofs_bottom, V.sub(1))]

    # DG-0 funciton for material
    V0 = FunctionSpace(mesh, ("DG", 0))
    mu_dg = Function(V0)
    lmbda_dg = Function(V0)
    mu_dg.interpolate(lambda x: np.full((1, x.shape[1]), mu))
    lmbda_dg.interpolate(lambda x: np.full((1, x.shape[1]), lmbda))
    sigma = sigma_func(mu, lmbda)

    # Pack mesh data for Nitsche solver
    contact = [(0, 1), (1, 0)]
    data = np.array([contact_bdy_1, contact_bdy_2], dtype=np.int32)
    offsets = np.array([0, 2], dtype=np.int32)
    surfaces = create_adjacencylist(data, offsets)

    # Function, TestFunction, TrialFunction and measures
    u = Function(V)
    v = ufl.TestFunction(V)
    dx = ufl.Measure("dx", domain=mesh, subdomain_data=domain_marker)
    ds = ufl.Measure("ds", domain=mesh, subdomain_data=facet_marker)

    # Create variational form without contact contributions
    F = ufl.inner(sigma(u), epsilon(v)) * dx

    # body forces
    F -= ufl.inner(t, v) * ds(top)

    problem_parameters = {"gamma": np.float64(E * 100), "theta": np.float64(-1), "friction": np.float64(0.3)}

    top_cells = domain_marker.find(1)
    # create initial guess

    def _u_initial(x):
        values = np.zeros((mesh.geometry.dim, x.shape[1]))
        values[-1] = -R / 100 - gap
        return values
    u.interpolate(_u_initial, top_cells)
    search_mode = [ContactMode.ClosestPoint, ContactMode.ClosestPoint]

    # Solve contact problem using Nitsche's method
    u, newton_its, krylov_iterations, solver_time = nitsche_unbiased(1, ufl_form=F,
                                                                     u=u, mu=mu_dg, lmbda=lmbda_dg, rhs_fns=[t],
                                                                     markers=[domain_marker, facet_marker],
                                                                     contact_data=(
                                                                         surfaces, contact), bcs=bcs,
                                                                     problem_parameters=problem_parameters,
                                                                     newton_options=newton_options,
                                                                     petsc_options=petsc_options,
                                                                     search_method=search_mode,
                                                                     outfile=None,
                                                                     fname=outname,
                                                                     quadrature_degree=args.q_degree,
                                                                     search_radius=np.float64(-1),
                                                                     order=args.order, simplex=simplex,
                                                                     pressure_function=_pressure,
                                                                     projection_coordinates=[
                                                                         (tdim - 1, -R), (tdim - 1, -R - 0.1)],
                                                                     coulomb=True)

    # Step 2: Frictional contact
    geometry = mesh.geometry.x[:].copy()

    u2 = Function(V)
    # Create variational form without contact contributions
    F = ufl.inner(sigma(u2), epsilon(v)) * dx

    # body forces
    t2 = Constant(mesh, ScalarType((0.0, -p)))
    F -= ufl.inner(t2, v) * ds(top)

    problem_parameters = {"gamma": np.float64(E * 1000), "theta": np.float64(1), "friction": np.float64(0.3)}

    ksp_tol = 1e-8
    # petsc_options = {
    #     "matptap_via": "scalable",
    #     "ksp_type": "cg",
    #     "ksp_rtol": ksp_tol,
    #     "ksp_atol": ksp_tol,
    #     "pc_type": "gamg",
    #     "pc_mg_levels": 3,
    #     "pc_mg_cycles": 1,   # 1 is v, 2 is w
    #     "mg_levels_ksp_type": "chebyshev",
    #     "mg_levels_pc_type": "jacobi",
    #     "pc_gamg_type": "agg",
    #     "pc_gamg_coarse_eq_limit": 100,
    #     "pc_gamg_agg_nsmooths": 1,
    #     "pc_gamg_threshold": 1e-3,
    #     "pc_gamg_square_graph": 2,
    #     "pc_gamg_reuse_interpolation": False,
    #     "ksp_initial_guess_nonzero": False,
    #     "ksp_norm_type": "unpreconditioned"
    # }
    petsc_options = {"ksp_type": "preonly", "pc_type": "lu"}
    newton_options = {"relaxation_parameter": 0.7,
                      "atol": newton_tol,
                      "rtol": newton_tol,
                      "convergence_criterion": "residual",
                      "max_it": 50,
                      "error_on_nonconvergence": True}
    symmetry_nodes = locate_entities(mesh, 0, lambda x: np.logical_and(np.isclose(x[0], 0), np.isclose(x[1], -R)))
    dofs_symmetry = locate_dofs_topological(V.sub(0), 0, symmetry_nodes)
    dofs_bottom = locate_dofs_topological(V, 1, facet_marker.find(bottom))
    bcs2 = [dirichletbc(Constant(mesh, ScalarType((0, 0))), dofs_bottom, V)]
    # Solve contact problem using Nitsche's method
    # update_geometry(u._cpp_object, mesh._cpp_object)
    # u2.x.array[:] = u.x.array[:]
    steps = 8
    contact_problem = create_contact_solver(ufl_form=F, u=u2, mu=mu_dg, lmbda=lmbda_dg,
                                            markers=[domain_marker, facet_marker],
                                            contact_data=(surfaces, contact),
                                            bcs=bcs2,
                                            problem_parameters=problem_parameters,
                                            newton_options=newton_options,
                                            petsc_options=petsc_options,
                                            search_method=search_mode,
                                            quadrature_degree=args.q_degree,
                                            search_radius=np.float64(0.5),
                                            coulomb=True, dt=1. / (steps + 2))

    contact_problem.u.x.array[:] = u.x.array[:]
    contact_problem.contact.update_submesh_geometry(contact_problem.u._cpp_object)
    # initialise vtx write
    vtx = VTXWriter(mesh.comm, "results/cylinders_coulomb.bp", [contact_problem.u])
    vtx.write(0)
    for i in range(1, steps + 1):
        for j in range(len(contact)):
            contact_problem.contact.create_distance_map(j)

        print(contact_problem.du.x.array[:])

        t2.value = ScalarType(((i) * 0.03 / steps, -p))
        n = contact_problem.solve()
        contact_problem.du.x.scatter_forward()
        contact_problem.u.x.array[:] += contact_problem.du.x.array[:]
        contact_problem.set_normals()
        contact_problem.contact.update_submesh_geometry(contact_problem.u._cpp_object)
        # take a fraction of du as initial guess
        # this is to ensure non-singular matrices in the case of no Dirichlet boundary
        contact_problem.du.x.array[:] = 0.1 * contact_problem.du.x.array[:]
        vtx.write((i + 1) / steps)

    vtx.close()
