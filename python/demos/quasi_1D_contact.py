# Copyright (C) 2023 Sarah Roggendorf
#
# SPDX-License-Identifier:    MIT

import argparse

import matplotlib.pyplot as plt

import numpy as np
import ufl
from dolfinx.io import XDMFFile
from dolfinx.fem import (Constant, Expression, Function, FunctionSpace, IntegralType,
                         VectorFunctionSpace, locate_dofs_topological, form,
                         assemble_scalar)
from dolfinx.graph import create_adjacencylist
from dolfinx.mesh import locate_entities
from mpi4py import MPI
from petsc4py.PETSc import ScalarType

from dolfinx_contact.helpers import (epsilon, sigma_func, lame_parameters)
from dolfinx_contact.meshing import (convert_mesh, 
                                     create_2D_rectangle_split)
import dolfinx_contact
from dolfinx_contact.cpp import ContactMode
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
    outname = "results/quasi_1D_simplex" if simplex else "results/quasi_1D_quads"
    fname = f"{mesh_dir}/quasi_1D_simplex" if simplex else f"{mesh_dir}/quasi_1D_quads"
    gap = 0.2
    create_2D_rectangle_split(filename=f"{fname}.msh", res=args.res, order=args.order, quads=not simplex, gap=gap)
    convert_mesh(fname, f"{fname}.xdmf", gdim=2)


    with XDMFFile(MPI.COMM_WORLD, f"{fname}.xdmf", "r") as xdmf:
        mesh = xdmf.read_mesh()
        domain_marker = xdmf.read_meshtags(mesh, name="cell_marker")
        tdim = mesh.topology.dim
        mesh.topology.create_connectivity(tdim - 1, tdim)
        facet_marker = xdmf.read_meshtags(mesh, name="facet_marker")

    dirichlet_bdy_1 = 2
    dirichlet_bdy_2 = 8
    contact_bdy_1 = 4
    contact_bdy_2 = 6

    disp_x = 0.2

    V = VectorFunctionSpace(mesh, ("CG", args.order))
    dirichlet_dofs1 = locate_dofs_topological(V, mesh.topology.dim - 1, facet_marker.find(dirichlet_bdy_1))
    L = 0.5
    H = 0.5
    dirichlet_nodes = locate_entities(mesh, 0, lambda x: np.logical_and(
            np.isclose(x[1], 0.5*H), np.logical_or(np.isclose(x[0], 2*L+gap-args.res/5), np.isclose(x[0], 2*L+gap-args.res/10))))
    print(dirichlet_nodes)
    dirichlet_dofs2 = locate_dofs_topological(V.sub(1), 0, dirichlet_nodes)
    # dirichlet_dofs2 = locate_dofs_topological(V, mesh.topology.dim - 1, facet_marker.find(dirichlet_bdy_2))
    bc_fns = [Constant(mesh, ScalarType((0, 0))), Constant(mesh, ScalarType(0.0))]

    bcs = ([(dirichlet_dofs1, -1), (dirichlet_dofs2, 1)], bc_fns)

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

    # Problem parameters
    E = 3
    nu = 0
    mu_func, lambda_func = lame_parameters(False)
    mu = mu_func(E, nu)
    lmbda = lambda_func(E, nu)
    sigma = sigma_func(mu, lmbda)

    def _f1(x):
        values = np.zeros((mesh.geometry.dim, x.shape[1]))
        values[0] = -disp_x*E*0.25*np.pi**2*np.sin(np.pi*x[0]/2)
        return values
    
    def _f2(x):
        values = np.zeros((mesh.geometry.dim, x.shape[1]))
        values[0] = -disp_x*E*0.25*np.pi**2*np.sin(np.pi*(x[0]-gap)/2)
        return values
    
    
    f = Function(V)
    cells_right = domain_marker.find(2)
    cells_left = domain_marker.find(1)
    f.interpolate(_f1, cells_left)
    f.interpolate(_f2, cells_right)
    # Create variational form without contact contributions
    F = ufl.inner(sigma(u), epsilon(v)) * dx
    # body forces
    F -= ufl.inner(f, v) * dx

    problem_parameters = {"mu": mu, "lambda": lmbda, "gamma": 1000*E, "theta": -1}

    # create initial guess
    def _u_initial(x):
        values = np.zeros((mesh.geometry.dim, x.shape[1]))
        values[0] = -0.1-gap
        return values
    u.interpolate(_u_initial, cells_right)


    search_mode = [ContactMode.ClosestPoint, ContactMode.ClosestPoint]
    # Solve contact problem using Nitsche's method
    u, newton_its, krylov_iterations, solver_time, contact, pn = nitsche_unbiased(1, ufl_form=F,
                                                                                  u=u, rhs_fns=[f],
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
                                                                                  search_radius=-1)
    
    def _exact1(x):
        values = np.zeros((mesh.geometry.dim, x.shape[1]))
        values[0] = -disp_x*np.sin(np.pi*x[0]/2)
        return values
    
    def _exact2(x):
        values = np.zeros((mesh.geometry.dim, x.shape[1]))
        values[0] = -disp_x*np.sin(np.pi*(x[0]-gap)/2) - gap
        return values
    
    
    exact = Function(V)
    exact.interpolate(_exact1, cells_left)
    exact.interpolate(_exact2, cells_right)

    with XDMFFile(mesh.comm, "results/u.xdmf", "w") as xdmf:
        xdmf.write_mesh(mesh)
        u.name = "u"
        xdmf.write_function(u)
    with XDMFFile(mesh.comm, "results/exact.xdmf", "w") as xdmf:
        xdmf.write_mesh(mesh)
        exact.name = "exact"
        xdmf.write_function(exact)

    exact.x.array[:] -= u.x.array[:]

    error_form = form(ufl.inner(ufl.grad(exact), ufl.grad(exact)) * dx)
    error = assemble_scalar(error_form)
    error = np.sqrt(error)
    error_form_l2 = form(ufl.inner(exact, exact) * dx)
    error_l2 = assemble_scalar(error_form_l2)
    error_l2 = np.sqrt(error_l2)
    nnodes = V.dofmap.index_map.size_global
    print(args.res, nnodes, error, error_l2)
    with XDMFFile(mesh.comm, "results/error.xdmf", "w") as xdmf:
        xdmf.write_mesh(mesh)
        exact.name = "error"
        xdmf.write_function(exact)