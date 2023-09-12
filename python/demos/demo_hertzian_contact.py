# Copyright (C) 2023 Sarah Roggendorf
#
# SPDX-License-Identifier:    MIT

import argparse

import numpy as np
import numpy.typing as npt
import ufl
from dolfinx.io import XDMFFile
from dolfinx.fem import (Constant, dirichletbc, Expression, Function, FunctionSpace,
                         VectorFunctionSpace, locate_dofs_topological)
from dolfinx.graph import adjacencylist
from dolfinx.geometry import bb_tree, compute_closest_entity
from dolfinx.mesh import Mesh

from mpi4py import MPI
from petsc4py.PETSc import ScalarType

from dolfinx_contact.helpers import (epsilon, sigma_func, lame_parameters)
from dolfinx_contact.meshing import (convert_mesh,
                                     create_circle_plane_mesh,
                                     create_halfdisk_plane_mesh,
                                     create_halfsphere_box_mesh)
from dolfinx_contact.cpp import ContactMode


from dolfinx_contact.unbiased.nitsche_unbiased import nitsche_unbiased


def closest_node_in_mesh(mesh: Mesh, point: npt.NDArray[np.float64]) -> npt.NDArray[np.int32]:
    points = np.reshape(point, (1, 3))
    bounding_box = bb_tree(mesh, 0)
    node = compute_closest_entity(bounding_box, bounding_box, mesh, points[0])
    return node


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
    parser.add_argument("--problem", default=1, type=int, dest="problem",
                        help="Which problem to solve: 1. Volume force, 2. Surface force",
                        choices=[1, 2])
    # Parse input arguments or set to defualt values
    args = parser.parse_args()
    threed = args.threed
    simplex = args.simplex
    problem = args.problem
    mesh_dir = "meshes"

    # Problem paramters
    R = 8  # 0.25
    L = 20  # 1.0
    H = 20  # 1.0
    if threed:
        load = 4 * 0.25 * np.pi * R**3 / 3.
    else:
        load = 0.25 * np.pi * R**2
    load = 2 * R * 0.625
    gap = 0.01

    # lame parameters
    E1 = 1e8  # 2.5
    E2 = 200  # 2.5
    nu1 = 0.0  # 0.25
    nu2 = 0.3  # 0.25
    mu_func, lambda_func = lame_parameters(True)
    mu1 = mu_func(E1, nu1)
    mu2 = mu_func(E2, nu2)
    lmbda1 = lambda_func(E1, nu1)
    lmbda2 = lambda_func(E2, nu2)
    Estar = E1 * E2 / (E2 * (1 - nu1**2) + E1 * (1 - nu2**2))

    if threed:
        if problem == 1:
            outname = "results/hertz1_3D_simplex"
            fname = f"{mesh_dir}/hertz1_3D_simplex"
            create_halfsphere_box_mesh(filename=f"{fname}.msh", res=args.res,
                                       order=args.order, r=R, H=H, L=L, W=L, gap=gap)
            neumann_bdy = 2
            contact_bdy_1 = 1
            contact_bdy_2 = 8
            dirichlet_bdy = 7
        else:
            outname = "results/hertz2_3D_simplex"
            fname = f"{mesh_dir}/hertz2_3D_simplex"
            create_halfsphere_box_mesh(filename=f"{fname}.msh", res=args.res,
                                       order=args.order, r=R, H=H, L=L, W=L, gap=gap)
            neumann_bdy = 2
            contact_bdy_1 = 1
            contact_bdy_2 = 8
            dirichlet_bdy = 7

        convert_mesh(fname, f"{fname}.xdmf", gdim=3)
        with XDMFFile(MPI.COMM_WORLD, f"{fname}.xdmf", "r") as xdmf:
            mesh = xdmf.read_mesh()
            domain_marker = xdmf.read_meshtags(mesh, name="cell_marker")
            tdim = mesh.topology.dim
            mesh.topology.create_connectivity(tdim - 1, tdim)
            facet_marker = xdmf.read_meshtags(mesh, name="facet_marker")

        V = VectorFunctionSpace(mesh, ("CG", args.order))

        node1 = closest_node_in_mesh(mesh, np.array([0.0, 0.0, 0.0], dtype=np.float64))
        node2 = closest_node_in_mesh(mesh, np.array([0.0, 0.0, -R / 2.0], dtype=np.float64))
        node3 = closest_node_in_mesh(mesh, np.array([0.0, R / 2.0, -R / 2.0], dtype=np.float64))
        dirichlet_nodes = np.array([node1, node2, node3], dtype=np.int32)
        dirichlet_dofs1 = locate_dofs_topological(V.sub(0), 0, dirichlet_nodes)
        dirichlet_dofs2 = locate_dofs_topological(V.sub(1), 0, dirichlet_nodes)
        dirichlet_dofs3 = locate_dofs_topological(V, mesh.topology.dim - 1, facet_marker.find(dirichlet_bdy))

        bcs = [dirichletbc(Constant(mesh, ScalarType(0.0)), dirichlet_dofs1, V.sub(0)),
               dirichletbc(Constant(mesh, ScalarType(0.0)), dirichlet_dofs2, V.sub(1)),
               dirichletbc(Constant(mesh, ScalarType((0.0, 0.0, 0.0))), dirichlet_dofs3, V)]

        if problem == 1:
            distributed_load = 3 * load / (4 * np.pi * R**3)
            f = Constant(mesh, ScalarType((0.0, -distributed_load)))
            t = Constant(mesh, ScalarType((0.0, 0.0)))
        else:
            distributed_load = load / (np.pi * R**2)
            f = Constant(mesh, ScalarType((0.0, 0.0, 0.0)))
            t = Constant(mesh, ScalarType((0.0, 0.0, -distributed_load)))

        a = np.cbrt(3.0 * load * R / (4 * Estar))
        force = 4 * a**3 * Estar / (3 * R)
        p0 = 3 * force / (2 * np.pi * a**2)

        def _pressure(x):
            vals = np.zeros(x.shape[1])
            for i in range(x.shape[1]):
                rsquared = x[0][i]**2 + x[1][i]**2
                if rsquared < a**2:
                    vals[i] = p0 * np.sqrt(1 - rsquared / a**2)
            return vals
    else:
        if problem == 1:
            outname = "results/hertz1_2D_simplex" if simplex else "results/hertz1_2D_quads"
            fname = f"{mesh_dir}/hertz1_2D_simplex" if simplex else f"{mesh_dir}/hertz1_2D_quads"
            create_circle_plane_mesh(f"{fname}.msh", not simplex, args.res, args.order, R, H, L, gap)
            contact_bdy_1 = 10
            contact_bdy_2 = 6
            dirichlet_bdy = 4
            neumann_bdy = 8
        else:
            outname = "results/hertz2_2D_simplex" if simplex else "results/hertz2_2D_quads"
            fname = f"{mesh_dir}/hertz2_2D_simplex" if simplex else f"{mesh_dir}/hertz2_2D_quads"
            create_halfdisk_plane_mesh(filename=f"{fname}.msh", res=args.res,
                                       order=args.order, quads=not simplex, r=R, H=H, L=L, gap=gap)
            contact_bdy_1 = 7
            contact_bdy_2 = 6
            dirichlet_bdy = 4
            neumann_bdy = 8

        convert_mesh(fname, f"{fname}.xdmf", gdim=2)
        with XDMFFile(MPI.COMM_WORLD, f"{fname}.xdmf", "r") as xdmf:
            mesh = xdmf.read_mesh()
            domain_marker = xdmf.read_meshtags(mesh, name="cell_marker")
            tdim = mesh.topology.dim
            mesh.topology.create_connectivity(tdim - 1, tdim)
            facet_marker = xdmf.read_meshtags(mesh, name="facet_marker")

        V = VectorFunctionSpace(mesh, ("CG", args.order))

        node1 = closest_node_in_mesh(mesh, np.array([0.0, -R / 2.5, 0.0], dtype=np.float64))
        node2 = closest_node_in_mesh(mesh, np.array([0.0, -R / 5.0, 0.0], dtype=np.float64))
        dirichlet_nodes = np.hstack([node1, node2])
        dirichlet_dofs1 = locate_dofs_topological(V.sub(0), 0, dirichlet_nodes)
        dirichlet_dofs2 = locate_dofs_topological(V, mesh.topology.dim - 1, facet_marker.find(dirichlet_bdy))

        bcs = [dirichletbc(Constant(mesh, ScalarType(0.0)), dirichlet_dofs1, V.sub(0)),
               dirichletbc(Constant(mesh, ScalarType((0, 0))), dirichlet_dofs2, V)]

        if problem == 1:
            distributed_load = load / (np.pi * R**2)
            f = Constant(mesh, ScalarType((0.0, -distributed_load)))
            t = Constant(mesh, ScalarType((0.0, 0.0)))
        else:
            distributed_load = load / (2 * R)
            f = Constant(mesh, ScalarType((0.0, 0.0)))
            t = Constant(mesh, ScalarType((0.0, -distributed_load)))

        a = 2 * np.sqrt(R * load / (np.pi * Estar))
        p0 = 2 * load / (np.pi * a)

        def _pressure(x):
            vals = np.zeros(x.shape[1])
            for i in range(x.shape[1]):
                if abs(x[0][i]) < a:
                    vals[i] = p0 * np.sqrt(1 - x[0][i]**2 / a**2)
            return vals

    # Solver options
    ksp_tol = 1e-8
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
        "pc_gamg_reuse_interpolation": False
    }

    # DG-0 funciton for material
    V0 = FunctionSpace(mesh, ("DG", 0))
    mu = Function(V0)
    lmbda = Function(V0)
    disk_cells = domain_marker.find(1)
    block_cells = domain_marker.find(2)
    mu.interpolate(lambda x: np.full((1, x.shape[1]), mu2), disk_cells)
    mu.interpolate(lambda x: np.full((1, x.shape[1]), mu1), block_cells)
    lmbda.interpolate(lambda x: np.full((1, x.shape[1]), lmbda2), disk_cells)
    lmbda.interpolate(lambda x: np.full((1, x.shape[1]), lmbda1), block_cells)

    sigma = sigma_func(mu, lmbda)
    # Set initial condition

    # Pack mesh data for Nitsche solver
    contact = [(0, 1), (0, 1)]
    data = np.array([contact_bdy_1, contact_bdy_2], dtype=np.int32)
    offsets = np.array([0, 2], dtype=np.int32)
    surfaces = adjacencylist(data, offsets)

    # Function, TestFunction, TrialFunction and measures
    u = Function(V)
    v = ufl.TestFunction(V)
    dx = ufl.Measure("dx", domain=mesh, subdomain_data=domain_marker)
    ds = ufl.Measure("ds", domain=mesh, subdomain_data=facet_marker)

    # Create variational form without contact contributions
    F = ufl.inner(sigma(u), epsilon(v)) * dx

    # body forces
    F -= ufl.inner(f, v) * dx(1) + ufl.inner(t, v) * ds(neumann_bdy)

    problem_parameters = {"gamma": np.float64(E2 * 100), "theta": np.float64(1)}

    # create initial guess
    def _u_initial(x):
        values = np.zeros((mesh.geometry.dim, x.shape[1]))
        values[-1] = -H / 100 - gap
        return values
    u.interpolate(_u_initial, disk_cells)
    search_mode = [ContactMode.ClosestPoint, ContactMode.ClosestPoint]

    # Solve contact problem using Nitsche's method
    u, newton_its, krylov_iterations, solver_time = nitsche_unbiased(1, ufl_form=F,
                                                                     u=u, mu=mu, lmbda=lmbda, rhs_fns=[f, t],
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
                                                                     projection_coordinates=(tdim - 1, -R))

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

        # xdmf.write_function(sigma_vm_h)

    # Create quadrature points for integration on facets
    # ct = mesh.topology.cell_type
    # x = []
    # for i in range(num_local):
    #     qps = contact.qp_phys(1, i)
    #     for pt in qps:
    #         x.append(pt[0])
    # print(len(x))

    # plt.figure()
    # plt.plot(x, pn[1], '*')
    # plt.xlabel('x')
    # plt.ylabel('p')

    # a = np.sqrt(8*load*R/(np.pi*Estar))
    # plt.xlim(-a-0.1, a+0.1)
    # r = np.linspace(-a, a, 100)
    # p0 = np.sqrt(load*Estar/(2*np.pi*R))
    # print(a, p0)
    # plt.grid()

    # p = p0 * np.sqrt(1 - r**2 / a**2)
    # plt.plot(r, p)
    # p_max = np.max(pn[1])
    # rel_err = np.abs(p_max - p0)/(max(p_max, p0))*100
    # print(p0, p_max, rel_err)
    # plt.savefig("contact_pressure.png")
