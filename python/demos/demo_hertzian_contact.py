# Copyright (C) 2023 Sarah Roggendorf
#
# SPDX-License-Identifier:    MIT

import argparse

import matplotlib.pyplot as plt

import numpy as np
import ufl
from dolfinx.io import XDMFFile
from dolfinx.fem import (Constant, Expression, Function, FunctionSpace, IntegralType,
                         VectorFunctionSpace, locate_dofs_topological)
from dolfinx.graph import create_adjacencylist
from dolfinx.mesh import locate_entities
from mpi4py import MPI
from petsc4py.PETSc import ScalarType

from dolfinx_contact.helpers import (epsilon, sigma_func, lame_parameters)
from dolfinx_contact.meshing import (convert_mesh,
                                     create_halfdisk_plane_mesh)
import dolfinx_contact


from dolfinx_contact.unbiased.nitsche_unbiased import nitsche_unbiased
if __name__ == "__main__":
    desc = "Example for verifying correctness of code"
    parser = argparse.ArgumentParser(description=desc,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--quadrature", default=5, type=int, dest="q_degree",
                        help="Quadrature degree used for contact integrals")
    parser.add_argument("--order", default=1, type=int, dest="order",
                        help="Order of mesh geometry", choices=[1, 2])
    _3D = parser.add_mutually_exclusive_group(required=False)
    _3D.add_argument('--3D', dest='threed', action='store_true',
                     help="Use 3D mesh", default=False)
    _simplex = parser.add_mutually_exclusive_group(required=False)
    _simplex.add_argument('--simplex', dest='simplex', action='store_true',
                          help="Use triangle/tet mesh", default=False)
    parser.add_argument("--res", default=0.1, type=np.float64, dest="res",
                        help="Mesh resolution")
    # Parse input arguments or set to defualt values
    args = parser.parse_args()
    # Current formulation uses bilateral contact
    threed = args.threed
    simplex = args.simplex
    mesh_dir = "meshes"
    if threed:
        pass
    else:
        outname = "results/problem2_2D_simplex" if simplex else "results/problem2_2D_quads"
        fname = f"{mesh_dir}/halfdisk" if simplex else f"{mesh_dir}/halfdisk"
        create_halfdisk_plane_mesh(filename=f"{fname}.msh", res=args.res, order=args.order)

        convert_mesh(fname, f"{fname}.xdmf", gdim=2)
        print("converted")

        with XDMFFile(MPI.COMM_WORLD, f"{fname}.xdmf", "r") as xdmf:
            mesh = xdmf.read_mesh()
            domain_marker = xdmf.read_meshtags(mesh, name="cell_marker")
            tdim = mesh.topology.dim
            mesh.topology.create_connectivity(tdim - 1, tdim)
            facet_marker = xdmf.read_meshtags(mesh, name="facet_marker")
        V = VectorFunctionSpace(mesh, ("CG", args.order))
        dirichlet_nodes = locate_entities(mesh, 0, lambda x: np.logical_and(
            np.isclose(x[0], 0), np.logical_or(np.isclose(x[1], 0), np.isclose(x[1], 0.1))))
        dirichlet_dofs1 = locate_dofs_topological(V.sub(0), 0, dirichlet_nodes)
        print(dirichlet_nodes)
        bc_fns = [Constant(mesh, ScalarType(0.0)), Constant(mesh, ScalarType((0.0, 0.0)))]
        dirichlet_dofs2 = locate_dofs_topological(V, mesh.topology.dim - 1, facet_marker.find(4))

        bcs = ([(dirichlet_dofs1, 0), (dirichlet_dofs2, -1)], bc_fns)
        contact_bdy_1 = 10
        contact_bdy_2 = 6
        f = Constant(mesh, ScalarType((0.0, -0.25)))  # body force

        # lame parameters
        E = 2.5
        nu = 0.25
        mu_func, lambda_func = lame_parameters(True)
        mu = mu_func(E, nu)
        lmbda = lambda_func(E, nu)
        sigma = sigma_func(mu, lmbda)

    # Solver options
    ksp_tol = 1e-10
    newton_tol = 1e-7
    newton_options = {"relaxation_parameter": 1,
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
    #     "pc_gamg_reuse_interpolation": False
    # }

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
    F -= ufl.inner(f, v) * dx(1)

    problem_parameters = {"mu": mu, "lambda": lmbda, "gamma": E * 15, "theta": 1}

    # Set initial condition
    Estar = E/(2*(1-nu**2))
    load = 0.25
    R = 0.25
    d = 4*load*0.25**2/Estar
    def _u_initial(x):
        values = np.zeros((mesh.geometry.dim, x.shape[1]))
        values[-1] = -d
        return values
    u.interpolate(_u_initial)

    # Solve contact problem using Nitsche's method
    u, newton_its, krylov_iterations, solver_time, contact, pn = nitsche_unbiased(8, ufl_form=F,
                                                                              u=u, rhs_fns=[f],
                                                                              markers=[domain_marker, facet_marker],
                                                                              contact_data=(surfaces, contact), bcs=bcs,
                                                                              problem_parameters=problem_parameters,
                                                                              newton_options=newton_options,
                                                                              petsc_options=petsc_options,
                                                                              outfile=None,
                                                                              fname=outname, raytracing=False,
                                                                              quadrature_degree=args.q_degree,
                                                                              search_radius=-1)

    sigma_dev = sigma(u) - (1 / 3) * ufl.tr(sigma(u)) * ufl.Identity(len(u))
    sigma_vm = ufl.sqrt((3 / 2) * ufl.inner(sigma_dev, sigma_dev))
    W = FunctionSpace(mesh, ("Discontinuous Lagrange", args.order - 1))
    sigma_vm_expr = Expression(sigma_vm, W.element.interpolation_points())
    sigma_vm_h = Function(W)
    sigma_vm_h.interpolate(sigma_vm_expr)
    sigma_vm_h.name = "vonMises"
    sigma_n_h = Function(W)
    n = ufl.FacetNormal(mesh)
    sigma_n = ufl.inner(sigma(u) * Constant(mesh, ScalarType((0.0, 1.0))), Constant(mesh, ScalarType((0.0, 1.0))))
    sigma_n_expr = Expression(sigma_n, W.element.interpolation_points())
    sigma_n_h.interpolate(sigma_n_expr)
    sigma_n_h.name = "p"
    u.name = "u"
    with XDMFFile(mesh.comm, "test_hertz.xdmf", "w") as xdmf:
        xdmf.write_mesh(mesh)
        u.name = "u"
        xdmf.write_function(u)
        # xdmf.write_function(sigma_vm_h)
        # xdmf.write_function(sigma_n_h)

    # Compuate integration entitites
    integration_entities, num_local = dolfinx_contact.compute_active_entities(mesh._cpp_object,
                                                                              facet_marker.find(contact_bdy_2),
                                                                              IntegralType.exterior_facet)
    integration_entities = integration_entities[:num_local]
    coeffs = dolfinx_contact.cpp.pack_coefficient_quadrature(
        sigma_n_h._cpp_object, args.q_degree, integration_entities)
    # Create quadrature points for integration on facets
    ct = mesh.topology.cell_type
    x = []
    for i in range(num_local):
        qps = contact.qp_phys(1, i)
        for pt in qps:
            x.append(pt[0])
    print(len(x))
    print(len(coeffs.reshape(-1)))

    plt.figure()
    plt.plot(x, pn[1], '*')
    plt.xlabel('x')
    plt.ylabel('p')
    plt.xlim(-0.25, 0.25)
    plt.ylim(0, 0.35)
    plt.grid()


    a = np.sqrt(d*R)
    r = np.linspace(-a, a, 100)
    p0 = np.sqrt(Estar*load*R)
    p = p0 * np.sqrt(1 - r**2 / a**2)
    plt.plot(r, p)
    print(p0, a)
    plt.savefig("contact_pressure.png")
