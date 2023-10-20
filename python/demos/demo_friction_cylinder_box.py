# Copyright (C) 2023 Sarah Roggendorf
#
# SPDX-License-Identifier:    MIT

import argparse

import numpy as np
import numpy.typing as npt
import ufl
from dolfinx.io import XDMFFile, VTXWriter
from dolfinx.fem import (assemble_scalar, Constant, dirichletbc, form,
                         Function, FunctionSpace,
                         VectorFunctionSpace, locate_dofs_topological)
from dolfinx.fem.petsc import set_bc
from dolfinx.geometry import bb_tree, compute_closest_entity
from dolfinx.graph import adjacencylist
from dolfinx.mesh import locate_entities, Mesh
from mpi4py import MPI
from petsc4py.PETSc import ScalarType

from dolfinx_contact.helpers import (epsilon, sigma_func, lame_parameters)
from dolfinx_contact.meshing import (convert_mesh,
                                     create_halfdisk_plane_mesh)
from dolfinx_contact.cpp import ContactMode


from dolfinx_contact.unbiased.contact_problem import create_contact_solver
from dolfinx_contact.output import ContactWriter

def closest_node_in_mesh(mesh: Mesh, point: npt.NDArray[np.float64]) -> npt.NDArray[np.int32]:
    points = np.reshape(point, (1, 3))
    bounding_box = bb_tree(mesh, 0)
    node = compute_closest_entity(bounding_box, bounding_box, mesh, points[0])
    return node

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
    L = 20
    H = 20
    gap = 0.01
    p = 0.625
    E = 200
    nu = 0.3
    Estar = E / (2 * (1 - nu**2))
    mu_func, lambda_func = lame_parameters(True)
    mu = mu_func(E, nu)
    lmbda = lambda_func(E, nu)

    # Create mesh
    name = "cylinder_box"
    outname = f"results/{name}_simplex" if simplex else f"results/{name}_quads"
    fname = f"{mesh_dir}/{name}_simplex" if simplex else f"{mesh_dir}/{name}_quads"
    create_halfdisk_plane_mesh(filename=f"{fname}.msh", res=args.res,
                                       order=args.order, quads=not simplex, r=R, H=H, L=L, gap=gap)
    convert_mesh(fname, f"{fname}.xdmf", gdim=2)
    with XDMFFile(MPI.COMM_WORLD, f"{fname}.xdmf", "r") as xdmf:
        mesh = xdmf.read_mesh()
        domain_marker = xdmf.read_meshtags(mesh, name="cell_marker")
        tdim = mesh.topology.dim
        mesh.topology.create_connectivity(tdim - 1, tdim)
        facet_marker = xdmf.read_meshtags(mesh, name="facet_marker")

    contact_bdy_1 = 7
    contact_bdy_2 = 6
    top = 8
    bottom = 4

    # Solver options
    ksp_tol = 1e-10
    newton_tol = 1e-7
    newton_options = {"snes_monitor": None, "snes_max_it": 50,
                    "snes_max_fail": 20, "snes_type": "newtonls",
                    "snes_linesearch_type": "basic",
                    "snes_linesearch_order": 1,
                    "snes_rtol": 1e-10, "snes_atol": 1e-10, "snes_view": None}
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

    node1 = closest_node_in_mesh(mesh, np.array([0.0, -R / 2.5, 0.0], dtype=np.float64))
    node2 = closest_node_in_mesh(mesh, np.array([0.0, -R / 5.0, 0.0], dtype=np.float64))
    dirichlet_nodes = np.hstack([node1, node2])
    dofs_symmetry = locate_dofs_topological(V.sub(0), 0, dirichlet_nodes)
    dofs_bottom = locate_dofs_topological(V, 1, facet_marker.find(bottom))
    dofs_top = locate_dofs_topological(V, 1, facet_marker.find(top))

    g_top = Constant(mesh, ScalarType((0.0, -0.1)))
    bcs = [dirichletbc(g, dofs_symmetry, V.sub(0)),
           dirichletbc(Constant(mesh, ScalarType((0.0, 0.0))), dofs_bottom, V),
           dirichletbc(g_top, dofs_top, V)]

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
    surfaces = adjacencylist(data, offsets)

    # Function, TestFunction, TrialFunction and measures
    u = Function(V)
    v = ufl.TestFunction(V)
    dx = ufl.Measure("dx", domain=mesh, subdomain_data=domain_marker)
    ds = ufl.Measure("ds", domain=mesh, subdomain_data=facet_marker)

    # Create variational form without contact contributions
    F = ufl.inner(sigma(u), epsilon(v)) * dx

    # body forces
    F -= ufl.inner(t, v) * ds(top)

    # Set up force postprocessing
    n = ufl.FacetNormal(mesh)
    ex = Constant(mesh, ScalarType((1.0, 0.0)))
    ey = Constant(mesh, ScalarType((0.0, 1.0)))
    Rx_form = form(ufl.inner(sigma(u) * n, ex) * ds(top))
    Ry_form = form(ufl.inner(sigma(u) * n, ey) * ds(top))

    def _tangent(x, p, a, c):
        vals = np.zeros(x.shape[1])
        for i in range(x.shape[1]):
            if abs(x[0][i]) <= c:
                vals[i] = fric * 4 * R * p / (np.pi * a**2) * (np.sqrt(a**2 - x[0][i]**2) - np.sqrt(c**2 - x[0][i]**2))
            elif abs(x[0][i] < a):
                vals[i] = fric * 4 * R * p / (np.pi * a**2) * (np.sqrt(a**2 - x[0][i]**2))
        return vals

    def _pressure(x, p0, a):
        vals = np.zeros(x.shape[1])
        for i in range(x.shape[1]):
            if abs(x[0][i]) < a:
                vals[i] = p0 * np.sqrt(1 - x[0][i]**2 / a**2)
        return vals

    problem_parameters = {"gamma": np.float64(E * 100 * args.order**2),
                          "theta": np.float64(1), "friction": np.float64(0.0)}

    top_cells = domain_marker.find(1)

    # u.interpolate(_u_initial, top_cells)
    search_mode = [ContactMode.ClosestPoint, ContactMode.ClosestPoint]

    # Solve contact problem using Nitsche's method
    steps1 = 4
    problem1 = create_contact_solver(ufl_form=F, u=u, mu=mu_dg, lmbda=lmbda_dg,
                                     markers=[domain_marker, facet_marker],
                                     contact_data=(surfaces, contact),
                                     bcs=bcs, problem_parameters=problem_parameters,
                                     newton_options=newton_options,
                                     petsc_options=petsc_options,
                                     search_method=search_mode,
                                     quadrature_degree=args.q_degree,
                                     search_radius=np.float64(1.0))

    problem1.update_friction_coefficient(0.0)
    h = problem1.h_surfaces()[1]
    # create initial guess

    def _u_initial(x):
        values = np.zeros((mesh.geometry.dim, x.shape[1]))
        values[-1] = -0.01 * h - gap
        return values

    problem1.du.interpolate(_u_initial, top_cells)

    writer = ContactWriter(mesh, problem1.contact, problem1.u, contact,
                           args.q_degree, search_mode, problem1.entities,
                           problem1.coeffs, args.order, simplex,
                           [(tdim - 1, 0), (tdim - 1, -R)],
                           f'{outname}')
    # initialise vtx writer
    vtx = VTXWriter(mesh.comm, f"{outname}.bp", [problem1.u], "bp4")
    vtx.write(0)
    newton_steps1 = []
    for i in range(steps1):

        for j in range(len(contact)):
            problem1.contact.create_distance_map(j)
        #val = -p * (i + 1) / steps1  # -0.2 / steps1  #
        g_top.value[1] = -0.3 / steps1
        t.value[1] = 0 #val
        print(f"Fricitionless part: Step {i+1} of {steps1}----------------------------------------------")
        # g_top.value[1] = val
        set_bc(problem1.du.vector, bcs)
        n = problem1.solve()
        newton_steps1.append(n)
        problem1.du.x.scatter_forward()
        problem1.u.x.array[:] += problem1.du.x.array[:]

        problem1.set_normals()
        # Compute forces
        # pr = abs(val)
        R_x = mesh.comm.allreduce(assemble_scalar(Rx_form), op=MPI.SUM)
        R_y = mesh.comm.allreduce(assemble_scalar(Ry_form), op=MPI.SUM)
        
        pr = abs(R_y / (2 * R))
        q = abs(R_x / (2 * R))
        load = pr * 2 * R  
        a = 2 * np.sqrt(R * load / (np.pi * Estar))
        p0 = 2 * load / (np.pi * a)
        print(pr, q)
        # print(val, 0)
        fric = 0.3
        c = a * np.sqrt(1 - 0 / (fric * pr))
        writer.write(i + 1, lambda x: _pressure(x, p0, a), lambda x: _tangent(x, pr, a, c))
        problem1.contact.update_submesh_geometry(problem1.u._cpp_object)
        problem1.du.x.array[:] = 0.1 * h * problem1.du.x.array[:]
        vtx.write(i + 1)

    # # Step 2: Frictional contact
    # geometry = mesh.geometry.x[:].copy()

    # u2 = Function(V)
    # # Create variational form without contact contributions
    # F = ufl.inner(sigma(u2), epsilon(v)) * dx

    # # body forces
    # t2 = Constant(mesh, ScalarType((0.0, -p)))
    # F -= ufl.inner(t2, v) * ds(top)

    # problem_parameters = {"gamma": np.float64(E * 1000), "theta": np.float64(1), "friction": np.float64(0.3)}

    ksp_tol = 1e-12
    petsc_options = {
        "matptap_via": "scalable",
        "ksp_type": "gmres",
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
    petsc_options = {"ksp_type": "preonly", "pc_type": "lu"}

    # newton_options = {"relaxation_parameter": 1.0,
    #                   "atol": newton_tol,
    #                   "rtol": newton_tol,
    #                   "convergence_criterion": "residual",
    #                   "max_it": 200,
    #                   "error_on_nonconvergence": True}
    # symmetry_nodes = locate_entities(mesh, 0, lambda x: np.logical_and(np.isclose(x[0], 0), np.isclose(x[1], -R)))
    # dofs_symmetry = locate_dofs_topological(V.sub(0), 0, symmetry_nodes)
    # dofs_bottom = locate_dofs_topological(V, 1, facet_marker.find(bottom))
    # bcs2 = [dirichletbc(Constant(mesh, ScalarType((0, 0))), dofs_bottom, V)]
    # # Solve contact problem using Nitsche's method
    # # update_geometry(u._cpp_object, mesh._cpp_object)
    # # u2.x.array[:] = u.x.array[:]
    def identifier(x):
        return np.logical_and(np.logical_and(x[0]<-0.5, x[0]>-1), np.logical_and(x[1]>-0.5, x[1]<0.5))
    constraint_nodes = locate_entities(mesh, 0, identifier)
    dofs_constraint = locate_dofs_topological(V.sub(1), 0, constraint_nodes)
    g_top = Constant(mesh, ScalarType((0.0, 0.0)))
    bcs = [dirichletbc(Constant(mesh, ScalarType((0.0, 0.0))), dofs_bottom, V), 
           dirichletbc(g, dofs_constraint, V.sub(1)),
           dirichletbc(g_top, dofs_top, V)]

    problem1.bcs = bcs
    problem1.update_friction_coefficient(0.3)
    problem1.update_nitsche_parameters(E * 100 * args.order**2, 1)
    problem1.coulomb = True
    problem1.petsc_options = petsc_options
    problem1.newton_options = newton_options
    steps2 = 8

    def _du_initial(x):
        values = np.zeros((mesh.geometry.dim, x.shape[1]))
        values[0] = 0.001 * h
        values[1] = 0
        return values
    problem1.du.x.array[:] = 0.1 * problem1.du.x.array[:]
    # problem1.du.interpolate(_du_initial, top_cells)
    # initialise vtx write
    # vtx = VTXWriter(mesh.comm, "results/cylinders_coulomb.bp", [problem1.u])
    # vtx.write(0)
    newton_steps2 = []
    # problem1.newton_options['rtol'] = 3e-2
    # problem1.newton_options['atol'] = 3e-2
    for i in range(steps2):
        for j in range(len(contact)):
            problem1.contact.create_distance_map(j)
        # g_top.value[0] = 0.1 / steps2
        print(f"Fricitional part: Step {i+1} of {steps2}----------------------------------------------")
        # print(problem1.du.x.array[:])
        set_bc(problem1.du.vector, bcs)

        #t.value[0] = 0.03 * (i + 1) / steps2
        g_top.value[0] = 0.025 / steps2
        n = problem1.solve()
        newton_steps2.append(n)
        problem1.du.x.scatter_forward()
        problem1.u.x.array[:] += problem1.du.x.array[:]
        problem1.set_normals()
        # Compute forces
        # pr = p
        R_x = mesh.comm.allreduce(assemble_scalar(Rx_form), op=MPI.SUM)
        R_y = mesh.comm.allreduce(assemble_scalar(Ry_form), op=MPI.SUM)
        pr = abs(R_y / (2 * R))
        # q = 0.03 * (i + 1) / steps2  # 
        q = abs(R_x / (2 * R))
        load = 2 * R * abs(pr) 
        a = 2 * np.sqrt(R * load / (np.pi * Estar))
        p0 = 2 * load / (np.pi * a)
        print(pr, q)
        fric = 0.3
        c = a * np.sqrt(1 - q / (fric * abs(pr)))
        writer.write(steps1 + i + 1, lambda x: _pressure(x, p0, a), lambda x: _tangent(x, abs(pr), a, c))
        problem1.contact.update_submesh_geometry(problem1.u._cpp_object)
        # take a fraction of du as initial guess
        # this is to ensure non-singular matrices in the case of no Dirichlet boundary
        problem1.du.x.array[:] = 0.1 * h * problem1.du.x.array[:]
        vtx.write(steps1 + 1 + i)

    vtx.close()

    print("Newton iterations: ")
    print(newton_steps1)
    print(newton_steps2)
