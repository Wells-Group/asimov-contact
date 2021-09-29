# Copyright (C) 2021 Jørgen S. Dokken and Sarah Roggendorf
#
# SPDX-License-Identifier:    MIT

import dolfinx
import dolfinx.io
import dolfinx.log
import numpy as np
import ufl
from mpi4py import MPI
from petsc4py import PETSc
from typing import Tuple
from dolfinx_contact.helpers import NonlinearPDE_SNESProblem, lame_parameters,\
    epsilon, sigma_func, rigid_motions_nullspace


def snes_solver(mesh: dolfinx.cpp.mesh.Mesh, mesh_data: Tuple[dolfinx.MeshTags, int, int], physical_parameters: dict,
                refinement: int = 0, g: float = 0.0, vertical_displacement: float = -0.1):
    (facet_marker, top_value, bottom_value) = mesh_data
    """
    Solving contact problem against a rigid plane with gap -g from y=0 using PETSc SNES solver
    """
    # write mesh and facet markers to xdmf
    with dolfinx.io.XDMFFile(MPI.COMM_WORLD, "results/mf_snes.xdmf", "w") as xdmf:
        xdmf.write_mesh(mesh)
        xdmf.write_meshtags(facet_marker)

    # function space and problem parameters
    V = dolfinx.VectorFunctionSpace(mesh, ("CG", 1))  # function space
    E = physical_parameters["E"]  # young's modulus
    nu = physical_parameters["nu"]  # poisson ratio
    mu_func, lambda_func = lame_parameters(physical_parameters["strain"])
    mu = mu_func(E, nu)
    lmbda = lambda_func(E, nu)
    sigma = sigma_func(mu, lmbda)

    # Functions for penalty term. Not used at the moment.
    # def gap(u): # Definition of gap function
    #     x = ufl.SpatialCoordinate(mesh)
    #     return x[1]+u[1]-g
    # def maculay(x): # Definition of Maculay bracket
    #     return (x+abs(x))/2

    # elasticity variational formulation no contact
    u = dolfinx.Function(V)
    v = ufl.TestFunction(V)
    dx = ufl.Measure("dx", domain=mesh)
    F = ufl.inner(sigma(u), epsilon(v)) * dx - \
        ufl.inner(dolfinx.Constant(mesh, [0, ] * mesh.geometry.dim), v) * dx

    # Stored strain energy density (linear elasticity model)    # penalty = 0
    # psi = 1/2*ufl.inner(sigma(u), epsilon(u))
    # Pi = psi*dx #+ 1/2*(penalty*E/h)*ufl.inner(maculay(-gap(u)),maculay(-gap(u)))*ds(1)

    # # Compute first variation of Pi (directional derivative about u in the direction of v)
    # # Yields same F as above if penalty = 0 and body force 0
    # F = ufl.derivative(Pi, u, v)

    # Dirichlet boundary conditions
    def _u_D(x):
        values = np.zeros((mesh.geometry.dim, x.shape[1]))
        values[mesh.geometry.dim - 1] = vertical_displacement
        return values
    u_D = dolfinx.Function(V)
    u_D.interpolate(_u_D)
    u_D.name = "u_D"
    u_D.x.scatter_forward()
    tdim = mesh.topology.dim
    dirichlet_dofs = dolfinx.fem.locate_dofs_topological(
        V, tdim - 1, facet_marker.indices[facet_marker.values == top_value])
    bc = dolfinx.DirichletBC(u_D, dirichlet_dofs)
    # bcs = [bc]

    # create nonlinear problem
    problem = NonlinearPDE_SNESProblem(F, u, bc)

    # Inequality constraints (contact constraints)
    # The displacement u must be such that the current configuration x+u
    # remains in the box [xmin = -inf,xmax = inf] x [ymin = -g,ymax = inf]
    # inf replaced by large number for implementation
    lims = np.zeros(2 * mesh.geometry.dim)
    for i in range(mesh.geometry.dim):
        lims[2 * i] = -1e7
        lims[2 * i + 1] = 1e7
    lims[-2] = -g

    def _constraint_u(x):
        values = np.zeros((mesh.geometry.dim, x.shape[1]))
        for i in range(mesh.geometry.dim):
            values[i] = lims[2 * i + 1] - x[i]
        return values

    def _constraint_l(x):
        values = np.zeros((mesh.geometry.dim, x.shape[1]))
        for i in range(mesh.geometry.dim):
            values[i] = lims[2 * i] - x[i]
        return values

    umax = dolfinx.Function(V)
    umax.interpolate(_constraint_u)
    umin = dolfinx.Function(V)
    umin.interpolate(_constraint_l)

    # Create semismooth Newton solver (SNES)
    b = dolfinx.cpp.la.create_vector(V.dofmap.index_map, V.dofmap.index_map_bs)
    J = dolfinx.cpp.fem.create_matrix(problem.a_comp._cpp_object)
    snes = PETSc.SNES().create()
    opts = PETSc.Options()
    opts["snes_monitor"] = None
    # opts["snes_view"] = None
    opts["snes_max_it"] = 50
    opts["snes_no_convergence_test"] = False
    opts["snes_max_fail"] = 10
    opts["snes_type"] = "vinewtonrsls"
    opts["snes_rtol"] = 1e-9
    opts["snes_atol"] = 1e-9
    snes.setFromOptions()
    snes.setFunction(problem.F, b)
    snes.setJacobian(problem.J, J)
    snes.setVariableBounds(umin.vector, umax.vector)
    null_space = rigid_motions_nullspace(V)
    J.setNearNullSpace(null_space)
    ksp = snes.ksp
    ksp.setOptionsPrefix("snes_ksp_")
    opts = PETSc.Options()
    option_prefix = ksp.getOptionsPrefix()
    # Cannot use GAMG, see: https://gitlab.com/petsc/petsc/-/issues/829
    opts[f"{option_prefix}ksp_type"] = "cg"
    opts[f"{option_prefix}ksp_rtol"] = 1.0e-5
    opts[f"{option_prefix}pc_type"] = "jacobi"

    # opts[f"{option_prefix}pc_type"] = "hypre"
    # opts[f"{option_prefix}pc_hypre_type"] = 'boomeramg'
    # opts[f"{option_prefix}pc_hypre_boomeramg_max_iter"] = 1
    # opts[f"{option_prefix}pc_hypre_boomeramg_cycle_type"] = "v"

    # opts[f"{option_prefix}ksp_view"] = None
    ksp.setFromOptions()

    def _u_initial(x):
        values = np.zeros((mesh.geometry.dim, x.shape[1]))
        values[-1] = -0.01 - g
        return values

    u.interpolate(_u_initial)
    # dolfinx.log.set_log_level(dolfinx.log.LogLevel.INFO)
    with dolfinx.common.Timer(f"{refinement} Solve SNES"):
        snes.solve(None, u.vector)
    u.x.scatter_forward()

    assert(snes.getConvergedReason() > 1)
    assert(snes.getConvergedReason() < 4)
    with dolfinx.io.XDMFFile(MPI.COMM_WORLD, f"results/u_snes_{refinement}.xdmf", "w") as xdmf:
        xdmf.write_mesh(mesh)
        xdmf.write_function(u)

    return u
