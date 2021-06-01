import dolfinx
import dolfinx.io
import numpy as np
import ufl
from mpi4py import MPI
from petsc4py import PETSc
import dolfinx_cuas.cpp as cuas

from helpers import (epsilon, lame_parameters, rigid_motions_nullspace,
                     sigma_func, R_minus)


def nitsche_rigid_surface(mesh, mesh_data, physical_parameters,
                          nitsche_parameters={"gamma": 1, "theta": 1},
                          vertical_displacement=-0.1, nitsche_bc=False):
    (facet_marker, top_value, bottom_value, surface_value, surface_bottom) = mesh_data

    # Nitche parameters and variables
    theta = nitsche_parameters["theta"]
    h = ufl.Circumradius(mesh)
    gamma = nitsche_parameters["gamma"] * physical_parameters["E"] / h
    n_vec = np.zeros(mesh.geometry.dim)
    n_vec[mesh.geometry.dim - 1] = 1
    # FIXME: more general definition of n_2 needed for surface that is not a horizontal rectangular box.
    n_2 = ufl.as_vector(n_vec)  # Normal of plane (projection onto other body)
    n = ufl.FacetNormal(mesh)

    V = dolfinx.VectorFunctionSpace(mesh, ("CG", 1))

    E = physical_parameters["E"]
    nu = physical_parameters["nu"]
    mu_func, lambda_func = lame_parameters(physical_parameters["strain"])
    mu = mu_func(E, nu)
    lmbda = lambda_func(E, nu)
    sigma = sigma_func(mu, lmbda)

    def sigma_n(v):
        # NOTE: Different normals, see summary paper
        return -ufl.dot(sigma(v) * n, n_2)

    # Mimicking the plane y=-g
    bottom_facets = facet_marker.indices[facet_marker.values == bottom_value]
    gdim = mesh.geometry.dim
    fdim = mesh.topology.dim - 1
    mesh_geometry = mesh.geometry.x
    contact = cuas.Contact(facet_marker, bottom_value, surface_value)
    contact.create_distance_map(0)
    lookup = contact.map_0_to_1()
    bottom_facets = contact.facet_0()
    master_bbox = dolfinx.cpp.geometry.BoundingBoxTree(mesh, fdim, bottom_facets)

    def gap(x):
        # gap = -x[mesh.geometry.dim - 1] - g
        dist_vec_array = np.zeros((gdim, x.shape[1]))
        for i in range(x.shape[1]):
            xi = x[:, i]
            facet, R = dolfinx.cpp.geometry.compute_closest_entity(master_bbox, xi, mesh, R=10)
            index = np.argwhere(np.array(bottom_facets) == facet)[0, 0]
            if np.isclose(R, 0):
                facet_2 = lookup.links(index)[0]
                facet2_geometry = dolfinx.cpp.mesh.entities_to_geometry(mesh, fdim, [facet_2], False)
                coords = mesh_geometry[facet2_geometry][0]
                dist_vec = dolfinx.cpp.geometry.compute_distance_gjk(coords, xi)
                dist_vec_array[:gdim, i] = -dist_vec[:gdim]
        return dist_vec_array

    g_vec = dolfinx.Function(V)
    g_vec.interpolate(gap)
    u = dolfinx.Function(V)
    v = ufl.TestFunction(V)
    # metadata = {"quadrature_degree": 5}
    dx = ufl.Measure("dx", domain=mesh)
    ds = ufl.Measure("ds", domain=mesh,  # metadata=metadata,
                     subdomain_data=facet_marker)
    a = ufl.inner(sigma(u), epsilon(v)) * dx
    L = ufl.inner(dolfinx.Constant(mesh, [0, ] * mesh.geometry.dim), v) * dx

    # # Derivation of one sided Nitsche with gap function
    F = a - theta / gamma * sigma_n(u) * sigma_n(v) * ds(bottom_value) - L
    F += 1 / gamma * R_minus(sigma_n(u) + gamma * (ufl.dot(g_vec, n_2) + ufl.dot(u, n_2))) * \
        (theta * sigma_n(v) + gamma * ufl.dot(v, n_2)) * ds(bottom_value)
    du = ufl.TrialFunction(V)
    q = sigma_n(u) + gamma * (ufl.dot(g_vec, n_2) + ufl.dot(u, n_2))
    J = ufl.inner(sigma(du), epsilon(v)) * ufl.dx - theta / gamma * sigma_n(du) * sigma_n(v) * ds(bottom_value)
    J += 1 / gamma * 0.5 * (1 - ufl.sign(q)) * (sigma_n(du) + gamma * ufl.dot(du, n_2)) * \
        (theta * sigma_n(v) + gamma * ufl.dot(v, n_2)) * ds(bottom_value)

    # # Nitsche for Dirichlet, another theta-scheme.
    # # https://doi.org/10.1016/j.cma.2018.05.024
    if nitsche_bc:
        disp_vec = np.zeros(mesh.geometry.dim)
        disp_vec[mesh.geometry.dim - 1] = vertical_displacement
        u_D = ufl.as_vector(disp_vec)
        F += - ufl.inner(sigma(u) * n, v) * ds(top_value)\
             - theta * ufl.inner(sigma(v) * n, u - u_D) * \
            ds(top_value) + gamma / h * ufl.inner(u - u_D, v) * ds(top_value)
        J += - ufl.inner(sigma(du) * n, v) * ds(top_value)\
            - theta * ufl.inner(sigma(v) * n, du) * \
            ds(top_value) + gamma / h * ufl.inner(du, v) * ds(top_value)
        # Nitsche bc for rigid plane
        disp_plane = np.zeros(mesh.geometry.dim)
        u_D_plane = ufl.as_vector(disp_plane)
        F += - ufl.inner(sigma(u) * n, v) * ds(surface_bottom)\
             - theta * ufl.inner(sigma(v) * n, u - u_D_plane) * \
            ds(surface_bottom) + gamma / h * ufl.inner(u - u_D_plane, v) * ds(surface_bottom)
        J += - ufl.inner(sigma(du) * n, v) * ds(surface_bottom)\
            - theta * ufl.inner(sigma(v) * n, du) * \
            ds(surface_bottom) + gamma / h * ufl.inner(du, v) * ds(surface_bottom)
        bcs = []
    else:
        # strong Dirichlet boundary conditions
        def _u_D(x):
            values = np.zeros((mesh.geometry.dim, x.shape[1]))
            values[mesh.geometry.dim - 1] = vertical_displacement
            return values
        tdim = mesh.topology.dim
        u_D = dolfinx.Function(V)
        u_D.interpolate(_u_D)
        u_D.name = "u_D"
        dolfinx.cpp.la.scatter_forward(u_D.x)
        dirichlet_dofs = dolfinx.fem.locate_dofs_topological(
            V, tdim - 1, facet_marker.indices[facet_marker.values == top_value])
        print(dirichlet_dofs)
        print(vertical_displacement)
        bc = dolfinx.DirichletBC(u_D, dirichlet_dofs)
        bcs = [bc]
        # Dirichlet boundary conditions for rigid plane
        dirichlet_dofs_plane = dolfinx.fem.locate_dofs_topological(
            V, tdim - 1, facet_marker.indices[facet_marker.values == surface_bottom])
        u_D_plane = dolfinx.Function(V)
        with u_D_plane.vector.localForm() as loc:
            loc.set(0)
        bc_plane = dolfinx.DirichletBC(u_D_plane, dirichlet_dofs_plane)
        bcs.append(bc_plane)

    # DEBUG: Write each step of Newton iterations
    # Create nonlinear problem and Newton solver
    # def form(self, x: PETSc.Vec):
    #     x.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
    #     self.i += 1
    #     xdmf.write_function(u, self.i)

    # setattr(dolfinx.fem.NonlinearProblem, "form", form)

    problem = dolfinx.fem.NonlinearProblem(F, u, bcs, J=J)
    # DEBUG: Write each step of Newton iterations
    # problem.i = 0
    # xdmf = dolfinx.io.XDMFFile(MPI.COMM_WORLD, "results/tmp_sol.xdmf", "w")
    # xdmf.write_mesh(mesh)

    solver = dolfinx.NewtonSolver(MPI.COMM_WORLD, problem)
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
        values[-1] = -0.01
        return values

    # Set initial_condition:
    u.interpolate(_u_initial)

    # Define solver and options
    ksp = solver.krylov_solver
    opts = PETSc.Options()
    option_prefix = ksp.getOptionsPrefix()
    # DEBUG: Use linear solver
    opts[f"{option_prefix}ksp_type"] = "preonly"
    opts[f"{option_prefix}pc_type"] = "lu"

    # FIXME: Need to figure out why this is not working
    # opts[f"{option_prefix}ksp_type"] = "cg"
    # opts[f"{option_prefix}pc_type"] = "gamg"
    # opts[f"{option_prefix}rtol"] = 1.0e-6
    # opts[f"{option_prefix}pc_gamg_coarse_eq_limit"] = 1000
    # opts[f"{option_prefix}mg_levels_ksp_type"] = "chebyshev"
    # opts[f"{option_prefix}mg_levels_pc_type"] = "jacobi"
    # opts[f"{option_prefix}mg_levels_esteig_ksp_type"] = "cg"
    # opts[f"{option_prefix}matptap_via"] = "scalable"
    # View solver options
    # opts[f"{option_prefix}ksp_view"] = None
    ksp.setFromOptions()

    # Solve non-linear problem
    dolfinx.log.set_log_level(dolfinx.log.LogLevel.INFO)
    with dolfinx.common.Timer("Solve Nitsche"):
        n, converged = solver.solve(u)
    dolfinx.cpp.la.scatter_forward(u.x)
    if solver.error_on_nonconvergence:
        assert(converged)
    print(f"{V.dofmap.index_map_bs*V.dofmap.index_map.size_global}, Number of interations: {n:d}")

    with dolfinx.io.XDMFFile(MPI.COMM_WORLD, "results/test_dist_vec.xdmf", "w") as xdmf:
        xdmf.write_mesh(mesh)
        xdmf.write_function(u)

    return u
