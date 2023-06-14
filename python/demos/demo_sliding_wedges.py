# Copyright (C) 2023 Sarah Roggendorf
#
# SPDX-License-Identifier:    MIT

import argparse

import numpy as np
import numpy.typing as npt
import ufl
from dolfinx.io import XDMFFile
from dolfinx.fem import (Constant, Expression, Function, FunctionSpace,
                         VectorFunctionSpace, locate_dofs_topological,
                         form, assemble_scalar)
from dolfinx.graph import create_adjacencylist
from dolfinx.mesh import locate_entities
from mpi4py import MPI
from petsc4py.PETSc import ScalarType

from dolfinx_contact import update_geometry
from dolfinx_contact.helpers import (epsilon, sigma_func, lame_parameters)
from dolfinx_contact.meshing import (convert_mesh,
                                     sliding_wedges)
from dolfinx_contact.cpp import ContactMode


from dolfinx_contact.unbiased.nitsche_unbiased import nitsche_unbiased

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
    fric = 0.1
    angle = np.arctan(0.1)

    # Create mesh
    outname = "results/sliding_wedges_simplex" if simplex else "results/sliding_wedges_quads"
    fname = f"{mesh_dir}/sliding_wedges_simplex" if simplex else f"{mesh_dir}/sliding_wedges_quads"
    sliding_wedges(f"{fname}.msh", not simplex, args.res, args.order, angle = angle)

    convert_mesh(fname, f"{fname}.xdmf", gdim=2)
    with XDMFFile(MPI.COMM_WORLD, f"{fname}.xdmf", "r") as xdmf:
        mesh = xdmf.read_mesh()
        domain_marker = xdmf.read_meshtags(mesh, name="cell_marker")
        tdim = mesh.topology.dim
        mesh.topology.create_connectivity(tdim - 1, tdim)
        facet_marker = xdmf.read_meshtags(mesh, name="facet_marker")


    contact_bdy_1 = 7
    contact_bdy_2 = 5
    neumann_bdy = 10
    dirichlet_bdy_1 = 9
    dirichlet_bdy_2 = 3

    V = VectorFunctionSpace(mesh, ("CG", args.order))
    # boundary conditions
    t = Constant(mesh, ScalarType((0.3, 0.0)))

    dirichlet_dofs_1 = locate_dofs_topological(V.sub(1), 1, facet_marker.find(dirichlet_bdy_1))
    dirichlet_dofs_2 = locate_dofs_topological(V, 1, facet_marker.find(dirichlet_bdy_2))

    bc_fns = [Constant(mesh, ScalarType((0.0))), Constant(mesh, ScalarType((0.0, 0.0)))]
    bcs = ([(dirichlet_dofs_1, 1), (dirichlet_dofs_2, -1)], bc_fns)

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
    F -= ufl.inner(t, v) * ds(neumann_bdy)

    problem_parameters = {"gamma": np.float64(E * 100), "theta": np.float64(-1), "friction": fric}

    search_mode = [ContactMode.ClosestPoint, ContactMode.ClosestPoint]
    

        # Solver options
    ksp_tol = 1e-10
    newton_tol = 1e-7
    newton_options = {"relaxation_parameter": 1.0,
                      "atol": newton_tol,
                      "rtol": newton_tol,
                      "convergence_criterion": "residual",
                      "max_it": 50,
                      "error_on_nonconvergence": True}
    petsc_options = {"ksp_type": "preonly", "pc_type": "lu"}
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


    def _pressure(x):
        vals = np.zeros(x.shape[1])
        return vals

    # Solve contact problem using Nitsche's method
    u, newton_its, krylov_iterations, solver_time= nitsche_unbiased(1, ufl_form=F,
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
                                                                     pressure_function= _pressure,
                                                                     projection_coordinates=[(tdim - 1, -R), (tdim - 1, -R - 0.1)],
                                                                     coulomb=True)
    
    n = ufl.FacetNormal(mesh)
    ex = Constant(mesh, ScalarType((1.0, 0.0)))
    ey = Constant(mesh, ScalarType((0.0, 1.0)))
    Rx_1form = form(ufl.inner(sigma(u)*n, ex) * ds(contact_bdy_1))
    Ry_1form = form(ufl.inner(sigma(u)*n, ey) * ds(contact_bdy_1))
    Rx_2form = form(ufl.inner(sigma(u)*n, ex) * ds(contact_bdy_2))
    Ry_2form = form(ufl.inner(sigma(u)*n, ey) * ds(contact_bdy_2))
    R_x1 = mesh.comm.allreduce(assemble_scalar(Rx_1form), op=MPI.SUM)
    R_y1 = mesh.comm.allreduce(assemble_scalar(Ry_1form), op=MPI.SUM)
    R_x2 = mesh.comm.allreduce(assemble_scalar(Rx_2form), op=MPI.SUM)
    R_y2 = mesh.comm.allreduce(assemble_scalar(Ry_2form), op=MPI.SUM)

    print("Rx/Ry", R_x1/R_y1, R_x2/R_y2, (fric + np.tan(angle))/(1 + fric*np.tan(angle)))
    sigma_dev = sigma(u) - (1 / 3) * ufl.tr(sigma(u)) * ufl.Identity(len(u))
    sigma_vm = ufl.sqrt((3 / 2) * ufl.inner(sigma_dev, sigma_dev))
    W = FunctionSpace(mesh, ("Discontinuous Lagrange", args.order - 1))
    sigma_vm_expr = Expression(sigma_vm, W.element.interpolation_points())
    sigma_vm_h = Function(W)
    sigma_vm_h.interpolate(sigma_vm_expr)
    sigma_vm_h.name = "vonMises"
    with XDMFFile(mesh.comm, f"{outname}_vonMises.xdmf", "w") as xdmf:
        xdmf.write_mesh(mesh)
        xdmf.write_function(sigma_vm_h)




    