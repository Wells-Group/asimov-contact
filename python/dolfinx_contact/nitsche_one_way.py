# Copyright (C) 2021 Jørgen S. Dokken and Sarah Roggendorf
#
# SPDX-License-Identifier:    MIT

from typing import Tuple

import dolfinx.common as _common
import dolfinx.fem as _fem
import dolfinx.io as _io
import dolfinx.log as _log
import dolfinx.mesh as _mesh
import dolfinx.nls as _nls
import numpy as np
import ufl
from petsc4py import PETSc as _PETSc

from dolfinx_contact.helpers import (R_minus, epsilon, lame_parameters,
                                     rigid_motions_nullspace, sigma_func)


def nitsche_one_way(mesh: _mesh.Mesh, mesh_data: Tuple[_mesh.MeshTags, int, int],
                    physical_parameters: dict, refinement: int = 0,
                    nitsche_parameters: dict = {"gamma": 1, "theta": 1, "s": 0}, g: float = 0.0,
                    vertical_displacement: float = -0.1, nitsche_bc: bool = False):
    (facet_marker, top_value, bottom_value) = mesh_data

    with _io.XDMFFile(mesh.comm, "results/mf_nitsche.xdmf", "w") as xdmf:
        xdmf.write_mesh(mesh)
        xdmf.write_meshtags(facet_marker)

    # Nitche parameters and variables
    theta = nitsche_parameters["theta"]
    # s = nitsche_parameters["s"]
    h = ufl.Circumradius(mesh)
    gamma = nitsche_parameters["gamma"] * physical_parameters["E"] / h
    n_vec = np.zeros(mesh.geometry.dim)
    n_vec[mesh.geometry.dim - 1] = -1
    n_2 = ufl.as_vector(n_vec)  # Normal of plane (projection onto other body)
    n = ufl.FacetNormal(mesh)

    V = _fem.VectorFunctionSpace(mesh, ("CG", 1))
    E = physical_parameters["E"]
    nu = physical_parameters["nu"]
    mu_func, lambda_func = lame_parameters(physical_parameters["strain"])
    mu = mu_func(E, nu)
    lmbda = lambda_func(E, nu)
    sigma = sigma_func(mu, lmbda)

    def sigma_n(v):
        # NOTE: Different normals, see summary paper
        return ufl.dot(sigma(v) * n, n_2)

    # Mimicking the plane y=-g
    x = ufl.SpatialCoordinate(mesh)
    gap = x[mesh.geometry.dim - 1] + g
    g_vec = [i for i in range(mesh.geometry.dim)]
    g_vec[mesh.geometry.dim - 1] = gap

    u = _fem.Function(V)
    v = ufl.TestFunction(V)
    # metadata = {"quadrature_degree": 5}
    dx = ufl.Measure("dx", domain=mesh)
    ds = ufl.Measure("ds", domain=mesh,  # metadata=metadata,
                     subdomain_data=facet_marker)
    a = ufl.inner(sigma(u), epsilon(v)) * dx
    L = ufl.inner(_fem.Constant(mesh, [0, ] * mesh.geometry.dim), v) * dx

    # Derivation of one sided Nitsche with gap function
    F = a - theta / gamma * sigma_n(u) * sigma_n(v) * ds(bottom_value) - L
    F += 1 / gamma * R_minus(sigma_n(u) + gamma * (gap - ufl.dot(u, n_2))) * \
        (theta * sigma_n(v) - gamma * ufl.dot(v, n_2)) * ds(bottom_value)
    du = ufl.TrialFunction(V)
    q = sigma_n(u) + gamma * (gap - ufl.dot(u, n_2))
    J = ufl.inner(sigma(du), epsilon(v)) * ufl.dx - theta / gamma * sigma_n(du) * sigma_n(v) * ds(bottom_value)
    J += 1 / gamma * 0.5 * (1 - ufl.sign(q)) * (sigma_n(du) - gamma * ufl.dot(du, n_2)) * \
        (theta * sigma_n(v) - gamma * ufl.dot(v, n_2)) * ds(bottom_value)

    # Nitsche for Dirichlet, another theta-scheme.
    # https://doi.org/10.1016/j.cma.2018.05.024
    if nitsche_bc:
        disp_vec = np.zeros(mesh.geometry.dim)
        disp_vec[mesh.geometry.dim - 1] = vertical_displacement
        u_D = ufl.as_vector(disp_vec)
        F += - ufl.inner(sigma(u) * n, v) * ds(top_value)\
             - theta * ufl.inner(sigma(v) * n, u - u_D) * \
            ds(top_value) + gamma / h * ufl.inner(u - u_D, v) * ds(top_value)
        bcs = []
        J += - ufl.inner(sigma(du) * n, v) * ds(top_value)\
            - theta * ufl.inner(sigma(v) * n, du) * \
            ds(top_value) + gamma / h * ufl.inner(du, v) * ds(top_value)
    else:
        # strong Dirichlet boundary conditions
        def _u_D(x):
            values = np.zeros((mesh.geometry.dim, x.shape[1]))
            values[mesh.geometry.dim - 1] = vertical_displacement
            return values
        u_D = _fem.Function(V)
        u_D.interpolate(_u_D)
        u_D.name = "u_D"
        u_D.x.scatter_forward()
        tdim = mesh.topology.dim
        dirichlet_dofs = _fem.locate_dofs_topological(
            V, tdim - 1, facet_marker.indices[facet_marker.values == top_value])
        bc = _fem.DirichletBC(u_D, dirichlet_dofs)
        bcs = [bc]

    # DEBUG: Write each step of Newton iterations
    # Create nonlinear problem and Newton solver
    # def form(self, x: _PETSc.Vec):
    #     x.ghostUpdate(addv=_PETSc.InsertMode.INSERT, mode=_PETSc.ScatterMode.FORWARD)
    #     self.i += 1
    #     xdmf.write_function(u, self.i)

    # setattr(_fem.NonlinearProblem, "form", form)

    problem = _fem.NonlinearProblem(F, u, bcs, J=J)
    # DEBUG: Write each step of Newton iterations
    # problem.i = 0
    # xdmf = _io.XDMFFile(mesh.comm, "results/tmp_sol.xdmf", "w")
    # xdmf.write_mesh(mesh)

    solver = _nls.NewtonSolver(mesh.comm, problem)
    null_space = rigid_motions_nullspace(V)
    solver.A.setNearNullSpace(null_space)

    # Set Newton solver options
    solver.atol = 1e-9
    solver.rtol = 1e-9
    solver.convergence_criterion = "incremental"
    solver.max_it = 50
    solver.error_on_nonconvergence = True
    solver.relaxation_parameter = 0.8

    def _u_initial(x):
        values = np.zeros((mesh.geometry.dim, x.shape[1]))
        values[-1] = -0.01 - g
        return values

    # Set initial_condition:
    u.interpolate(_u_initial)

    # Define solver and options
    ksp = solver.krylov_solver
    opts = _PETSc.Options()
    option_prefix = ksp.getOptionsPrefix()
    # DEBUG: Use linear solver
    # opts[f"{option_prefix}ksp_type"] = "preonly"
    # opts[f"{option_prefix}pc_type"] = "lu"

    opts[f"{option_prefix}ksp_type"] = "cg"
    opts[f"{option_prefix}pc_type"] = "gamg"
    opts[f"{option_prefix}rtol"] = 1.0e-6
    opts[f"{option_prefix}pc_gamg_coarse_eq_limit"] = 1000
    opts[f"{option_prefix}mg_levels_ksp_type"] = "chebyshev"
    opts[f"{option_prefix}mg_levels_pc_type"] = "jacobi"
    opts[f"{option_prefix}mg_levels_esteig_ksp_type"] = "cg"
    opts[f"{option_prefix}matptap_via"] = "scalable"
    # View solver options
    # opts[f"{option_prefix}ksp_view"] = None
    ksp.setFromOptions()

    # Solve non-linear problem
    _log.set_log_level(_log.LogLevel.INFO)
    with _common.Timer(f"{refinement} Solve Nitsche"):
        n, converged = solver.solve(u)
    u.x.scatter_forward()
    if solver.error_on_nonconvergence:
        assert(converged)
    print(f"{V.dofmap.index_map_bs*V.dofmap.index_map.size_global}, Number of interations: {n:d}")

    with _io.XDMFFile(mesh.comm, f"results/u_nitsche_{refinement}.xdmf", "w") as xdmf:
        xdmf.write_mesh(mesh)
        xdmf.write_function(u)

    return u
