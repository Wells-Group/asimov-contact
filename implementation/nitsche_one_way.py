import dolfinx
import dolfinx.io
import numpy as np
import ufl
from petsc4py import PETSc
from mpi4py import MPI

from helpers import epsilon, lame_parameters, sigma_func


def R_minus(x):
    abs_x = abs(x)
    return 0.5 * (x - abs_x)


def ball_projection(x, s):
    dim = x.geometric_dimension()
    abs_x = ufl.sqrt(sum([x[i]**2 for i in range(dim)]))
    return ufl.conditional(ufl.le(abs_x, s), x, s * x / abs_x)


def nitsche_one_way(mesh, mesh_data, physical_parameters, refinement=0,
                    nitsche_parameters={"gamma": 1, "theta": 1, "s": 0}, g=0.0,
                    vertical_displacement=-0.1, nitsche_bc=False):
    (facet_marker, top_value, bottom_value) = mesh_data

    with dolfinx.io.XDMFFile(MPI.COMM_WORLD, "results/mf_nitsche.xdmf", "w") as xdmf:
        xdmf.write_mesh(mesh)
        xdmf.write_meshtags(facet_marker)

    # Nitche parameters and variables
    theta = nitsche_parameters["theta"]
    # s = nitsche_parameters["s"]
    h = ufl.Circumradius(mesh)
    gamma = physical_parameters["E"] * nitsche_parameters["gamma"] / h
    # n = ufl.FacetNormal(mesh)
    n = ufl.as_vector((0, -1))  # Normal of plane
    V = dolfinx.VectorFunctionSpace(mesh, ("CG", 1))
    E = physical_parameters["E"]
    nu = physical_parameters["nu"]
    mu_func, lambda_func = lame_parameters(physical_parameters["strain"])
    mu = mu_func(E, nu)
    lmbda = lambda_func(E, nu)
    sigma = sigma_func(mu, lmbda)

    def sigma_n(v):
        return ufl.dot(sigma(v) * n, n)

    # def tangential_proj(u):
    #     """
    #     See for instance:
    #     https://doi.org/10.1023/A:1022235512626
    #     """
    #     return (ufl.Identity(u.ufl_shape[0]) - ufl.outer(n, n)) * u

    # Mimicking the plane y=-g
    x = ufl.SpatialCoordinate(mesh)
    gap = -x[1] - g

    u = dolfinx.Function(V)
    v = ufl.TestFunction(V)
    dx = ufl.Measure("dx", domain=mesh)
    ds = ufl.Measure("ds", domain=mesh,
                     subdomain_data=facet_marker, subdomain_id=bottom_value)
    a = ufl.inner(sigma(u), epsilon(v)) * dx
    L = ufl.inner(dolfinx.Constant(mesh, (0, 0)), v) * dx

    # Nitsche for contact (with Friction).
    # NOTE: Differs from unilateral contact even in the case of s=0!
    # F -= theta / gamma * sigma_n(u) * sigma_n(v) * ds(2)
    # F += 1 / gamma * R_minus(sigma_n(u) + gamma * (gap - ufl.dot(u, n))) * \
    #     (theta * sigma_n(v) - gamma * ufl.dot(v, n)) * ds(2)
    # F -= theta / gamma * ufl.dot(tangential_proj(u), tangential_proj(v)) * ds(2)
    # F += 1 / gamma * ufl.dot(ball_projection(tangential_proj(u) - gamma * tangential_proj(u), s),
    #                         theta * tangential_proj(v) - gamma * tangential_proj(v)) * ds(2)

    def An(theta, gamma): return a - theta / \
        gamma * sigma_n(u) * sigma_n(v) * ds

    def Pn(v, theta, gamma): return theta * sigma_n(v) - gamma * ufl.dot(v, n)
    F = An(theta, gamma) + 1 / gamma * R_minus(sigma_n(u) - gamma
                                               * ufl.dot(u - ufl.as_vector((0, gap)), n)) * Pn(v, theta, gamma) * ds - L
    # F -= theta / gamma * sigma_n(u) * sigma_n(v) * ds(2)
    # F += 1 / gamma * R_minus(sigma_n(u) - gamma * (ufl.dot(u, n) - g))* (theta * sigma_n(v) - gamma * ufl.dot(v, n)) * ds(2)

    # Nitsche for Dirichlet, another theta-scheme.
    # https://doi.org/10.1016/j.cma.2018.05.024
    # Ultimately, it might make sense to use the same theta as for contact. But we keep things separate for now.
    if nitsche_bc:
        u_D = ufl.as_vector((0, vertical_displacement))
        n_facet = ufl.FacetNormal(mesh)
        gamma_2 = 1000
        theta_2 = 1  # 1 symmetric, -1 skew symmetric
        F += - ufl.inner(sigma(u) * n_facet, v) * ds(1)\
             - theta_2 * ufl.inner(sigma(v) * n_facet, u - u_D) * \
            ds(1) + gamma_2 / h * ufl.inner(u - u_D, v) * ds(1)
        bcs = []
    else:
        # strong Dirichlet boundary conditions
        def _u_D(x):
            values = np.zeros((mesh.geometry.dim, x.shape[1]))
            values[0] = 0
            values[1] = vertical_displacement
            return values
        u_D = dolfinx.Function(V)
        u_D.interpolate(_u_D)
        u_D.name = "u_D"
        dolfinx.cpp.la.scatter_forward(u_D.x)
        tdim = mesh.topology.dim
        dirichlet_dofs = dolfinx.fem.locate_dofs_topological(
            V, tdim - 1, facet_marker.indices[facet_marker.values == top_value])
        bc = dolfinx.DirichletBC(u_D, dirichlet_dofs)
        bcs = [bc]

    # Create nonlinear problem and Newton solver
    problem = dolfinx.fem.NonlinearProblem(F, u, bcs)
    solver = dolfinx.NewtonSolver(MPI.COMM_WORLD, problem)

    # Set Newton solver options
    solver.atol = 1e-6
    solver.rtol = 1e-6
    solver.convergence_criterion = "incremental"
    solver.max_it = 50

    def _u_initial(x):
        values = np.zeros((mesh.geometry.dim, x.shape[1]))
        values[0] = 0
        values[1] = -0.1 * x[1]
        return values
    # Set initial_condition:
    u.interpolate(_u_initial)
    ksp = solver.krylov_solver
    opts = PETSc.Options()
    option_prefix = ksp.getOptionsPrefix()
    opts[f"{option_prefix}ksp_type"] = "cg"
    opts[f"{option_prefix}pc_type"] = "gamg"
    opts[f"{option_prefix}rtol"] = 1.0e-8
    opts[f"{option_prefix}pc_gamg_coarse_eq_limit"] = 1000
    opts[f"{option_prefix}mg_levels_ksp_type"] = "chebyshev"
    opts[f"{option_prefix}mg_levels_pc_type"] = "jacobi"
    opts[f"{option_prefix}mg_levels_esteig_ksp_type"] = "cg"
    opts[f"{option_prefix}matptap_via"] = "scalable"
    opts[f"{option_prefix}ksp_view"] = None
    ksp.setFromOptions()

    # Solve non-linear problem
    dolfinx.log.set_log_level(dolfinx.log.LogLevel.INFO)
    with dolfinx.common.Timer(f"{refinement} Solve Nitsche"):
        n, converged = solver.solve(u)
    dolfinx.cpp.la.scatter_forward(u.x)
    assert(converged)
    print(f"Number of interations: {n:d}")

    with dolfinx.io.XDMFFile(MPI.COMM_WORLD, f"results/u_nitsche_{refinement}.xdmf", "w") as xdmf:
        xdmf.write_mesh(mesh)
        xdmf.write_function(u)

    return u
