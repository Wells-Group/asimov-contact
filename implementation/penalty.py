import dolfinx
import dolfinx.io
import numpy as np
import ufl
from mpi4py import MPI
from petsc4py import PETSc

from helpers import (epsilon, lame_parameters, rigid_motions_nullspace,
                     sigma_func, R_minus)


def penalty(mesh, mesh_data, physical_parameters, refinement=0,
            nitsche_parameters={"gamma": 1, "theta": 1, "s": 0}, g=0.0,
            vertical_displacement=-0.1, nitsche_bc=False, initGuess=None, load_step=0):
    (facet_marker, top_value, bottom_value) = mesh_data

    with dolfinx.io.XDMFFile(MPI.COMM_WORLD, "results/mf_nitsche.xdmf", "w") as xdmf:
        xdmf.write_mesh(mesh)
        xdmf.write_meshtags(facet_marker)

    # Nitche parameters and variables
    theta = nitsche_parameters["theta"]
    # s = nitsche_parameters["s"]
    h = ufl.Circumradius(mesh)
    gamma = nitsche_parameters["gamma"] * physical_parameters["E"] / h
    n_vec = np.zeros(mesh.geometry.dim)
    n_vec[mesh.geometry.dim - 1] = 1
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
    x = ufl.SpatialCoordinate(mesh)
    gap = x[mesh.geometry.dim - 1] + g
    g_vec = [i for i in range(mesh.geometry.dim)]
    g_vec[mesh.geometry.dim - 1] = gap

    u = dolfinx.Function(V)
    v = ufl.TestFunction(V)
    # metadata = {"quadrature_degree": 5}
    dx = ufl.Measure("dx", domain=mesh)
    ds = ufl.Measure("ds", domain=mesh,  # metadata=metadata,
                     subdomain_data=facet_marker)
    a = ufl.inner(sigma(u), epsilon(v)) * dx
    L = ufl.inner(dolfinx.Constant(mesh, [0, ] * mesh.geometry.dim), v) * dx

    # Derivation of one sided Nitsche with gap function
    F = a - L
    F += gamma * R_minus((gap + ufl.dot(u, n_2))) * \
        (ufl.dot(v, n_2)) * ds(bottom_value)
    du = ufl.TrialFunction(V)
    q = (gap + ufl.dot(u, n_2))
    J = ufl.inner(sigma(du), epsilon(v)) * ufl.dx
    J += gamma * 0.5 * (1 - ufl.sign(q)) * (ufl.dot(du, n_2)) * \
        (ufl.dot(v, n_2)) * ds(bottom_value)

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
        u_D = dolfinx.Function(V)
        u_D.interpolate(_u_D)
        u_D.name = "u_D"
        u_D.x.scatter_forward()
        tdim = mesh.topology.dim
        dirichlet_dofs = dolfinx.fem.locate_dofs_topological(
            V, tdim - 1, facet_marker.indices[facet_marker.values == top_value])
        bc = dolfinx.DirichletBC(u_D, dirichlet_dofs)
        bcs = [bc]

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
    solver.convergence_criterion = "residual"
    solver.max_it = 50
    solver.error_on_nonconvergence = True
    solver.relaxation_parameter = 0.8

    if initGuess is None:
        # def _u_initial(x):
        #     values = np.zeros((mesh.geometry.dim, x.shape[1]))
        #     values[-1] = -0.01
        #     return values
        # # Set initial_condition:
        # u.interpolate(_u_initial)
        pass
    else:
        u.x.array[:] = initGuess.x.array
        # u.x.scatter_forward()
    print(f"norm of initial guess {np.linalg.norm(u.x.array)}")
    # Define solver and options
    ksp = solver.krylov_solver
    opts = PETSc.Options()
    option_prefix = ksp.getOptionsPrefix()
    # DEBUG: Use linear solver
    # opts[f"{option_prefix}ksp_type"] = "preonly"
    # opts[f"{option_prefix}pc_type"] = "lu"

    # KSP Object: (nls_solve_) 4 MPI processes
    #   type: cg
    #   maximum iterations=10000, initial guess is zero
    #   tolerances:  relative=1e-05, absolute=1e-50, divergence=10000.
    #   left preconditioning
    #   using PRECONDITIONED norm type for convergence test
    # PC Object: (nls_solve_) 4 MPI processes
    #   type: gamg
    #     type is MULTIPLICATIVE, levels=4 cycles=v
    #       Cycles per PCApply=1
    #       Using externally compute Galerkin coarse grid matrices
    #       GAMG specific options
    #         Threshold for dropping small values in graph on each level =   0.01   0.01   0.01   0.01
    #         Threshold scaling factor for each level not specified = 1.
    #         AGG specific options
    #           Symmetric graph true
    #           Number of levels to square graph 2
    #           Number smoothing steps 1
    #         Complexity:    grid = 1.53
    #   Coarse grid solver -- level -------------------------------
    #     KSP Object: (nls_solve_mg_coarse_) 4 MPI processes
    #       type: preonly
    #       maximum iterations=10000, initial guess is zero
    #       tolerances:  relative=1e-05, absolute=1e-50, divergence=10000.
    #       left preconditioning
    #       using NONE norm type for convergence test
    #     PC Object: (nls_solve_mg_coarse_) 4 MPI processes
    #       type: bjacobi
    #         number of blocks = 4
    #         Local solver information for first block is in the following KSP and PC objects on rank 0:
    #         Use -nls_solve_mg_coarse_ksp_view ::ascii_info_detail to display information for all blocks
    #       KSP Object: (nls_solve_mg_coarse_sub_) 1 MPI processes
    #         type: preonly
    #         maximum iterations=1, initial guess is zero
    #         tolerances:  relative=1e-05, absolute=1e-50, divergence=10000.
    #         left preconditioning
    #         using NONE norm type for convergence test
    #       PC Object: (nls_solve_mg_coarse_sub_) 1 MPI processes
    #         type: lu
    #           out-of-place factorization
    #           tolerance for zero pivot 2.22045e-14
    #           using diagonal shift on blocks to prevent zero pivot [INBLOCKS]
    #           matrix ordering: nd
    #           factor fill ratio given 5., needed 1.
    #             Factored matrix follows:
    #               Mat Object: 1 MPI processes
    #                 type: seqaij
    #                 rows=294, cols=294, bs=6
    #                 package used to perform factorization: petsc
    #                 total: nonzeros=86436, allocated nonzeros=86436
    #                   using I-node routines: found 59 nodes, limit used is 5
    #         linear system matrix = precond matrix:
    #         Mat Object: (nls_solve_mg_coarse_sub_) 1 MPI processes
    #           type: seqaij
    #           rows=294, cols=294, bs=6
    #           total: nonzeros=86436, allocated nonzeros=86436
    #           total number of mallocs used during MatSetValues calls=0
    #             using I-node routines: found 59 nodes, limit used is 5
    #       linear system matrix = precond matrix:
    #       Mat Object: 4 MPI processes
    #         type: mpiaij
    #         rows=294, cols=294, bs=6
    #         total: nonzeros=86436, allocated nonzeros=86436
    #         total number of mallocs used during MatSetValues calls=0
    #           using I-node (on process 0) routines: found 59 nodes, limit used is 5
    #   Down solver (pre-smoother) on level 1 -------------------------------
    #     KSP Object: (nls_solve_mg_levels_1_) 4 MPI processes
    #       type: chebyshev
    #         eigenvalue estimates used:  min = 0.14791, max = 1.62702
    #         eigenvalues estimate via cg min 0.014584, max 1.4791
    #         eigenvalues estimated using cg with translations  [0. 0.1; 0. 1.1]
    #         KSP Object: (nls_solve_mg_levels_1_esteig_) 4 MPI processes
    #           type: cg
    #           maximum iterations=10, initial guess is zero
    #           tolerances:  relative=1e-12, absolute=1e-50, divergence=10000.
    #           left preconditioning
    #           using PRECONDITIONED norm type for convergence test
    #         estimating eigenvalues using noisy right hand side
    #       maximum iterations=2, nonzero initial guess
    #       tolerances:  relative=1e-05, absolute=1e-50, divergence=10000.
    #       left preconditioning
    #       using NONE norm type for convergence test
    #     PC Object: (nls_solve_mg_levels_1_) 4 MPI processes
    #       type: sor
    #         type = local_symmetric, iterations = 1, local iterations = 1, omega = 1.
    #       linear system matrix = precond matrix:
    #       Mat Object: 4 MPI processes
    #         type: mpiaij
    #         rows=2202, cols=2202, bs=6
    #         total: nonzeros=1248228, allocated nonzeros=1248228
    #         total number of mallocs used during MatSetValues calls=0
    #           using nonscalable MatPtAP() implementation
    #           using I-node (on process 0) routines: found 134 nodes, limit used is 5
    #   Up solver (post-smoother) same as down solver (pre-smoother)
    #   Down solver (pre-smoother) on level 2 -------------------------------
    #     KSP Object: (nls_solve_mg_levels_2_) 4 MPI processes
    #       type: chebyshev
    #         eigenvalue estimates used:  min = 0.151576, max = 1.66733
    #         eigenvalues estimate via cg min 0.0282551, max 1.51576
    #         eigenvalues estimated using cg with translations  [0. 0.1; 0. 1.1]
    #         KSP Object: (nls_solve_mg_levels_2_esteig_) 4 MPI processes
    #           type: cg
    #           maximum iterations=10, initial guess is zero
    #           tolerances:  relative=1e-12, absolute=1e-50, divergence=10000.
    #           left preconditioning
    #           using PRECONDITIONED norm type for convergence test
    #         estimating eigenvalues using noisy right hand side
    #       maximum iterations=2, nonzero initial guess
    #       tolerances:  relative=1e-05, absolute=1e-50, divergence=10000.
    #       left preconditioning
    #       using NONE norm type for convergence test
    #     PC Object: (nls_solve_mg_levels_2_) 4 MPI processes
    #       type: sor
    #         type = local_symmetric, iterations = 1, local iterations = 1, omega = 1.
    #       linear system matrix = precond matrix:
    #       Mat Object: 4 MPI processes
    #         type: mpiaij
    #         rows=81726, cols=81726, bs=6
    #         total: nonzeros=27975060, allocated nonzeros=27975060
    #         total number of mallocs used during MatSetValues calls=0
    #           using nonscalable MatPtAP() implementation
    #           using I-node (on process 0) routines: found 6316 nodes, limit used is 5
    #   Up solver (post-smoother) same as down solver (pre-smoother)
    #   Down solver (pre-smoother) on level 3 -------------------------------
    #     KSP Object: (nls_solve_mg_levels_3_) 4 MPI processes
    #       type: chebyshev
    #         eigenvalue estimates used:  min = 0.169376, max = 1.86313
    #         eigenvalues estimate via cg min 0.0354186, max 1.69376
    #         eigenvalues estimated using cg with translations  [0. 0.1; 0. 1.1]
    #         KSP Object: (nls_solve_mg_levels_3_esteig_) 4 MPI processes
    #           type: cg
    #           maximum iterations=10, initial guess is zero
    #           tolerances:  relative=1e-12, absolute=1e-50, divergence=10000.
    #           left preconditioning
    #           using PRECONDITIONED norm type for convergence test
    #         estimating eigenvalues using noisy right hand side
    #       maximum iterations=2, nonzero initial guess
    #       tolerances:  relative=1e-05, absolute=1e-50, divergence=10000.
    #       left preconditioning
    #       using NONE norm type for convergence test
    #     PC Object: (nls_solve_mg_levels_3_) 4 MPI processes
    #       type: sor
    #         type = local_symmetric, iterations = 1, local iterations = 1, omega = 1.
    #       linear system matrix = precond matrix:
    #       Mat Object: 4 MPI processes
    #         type: mpiaij
    #         rows=1241136, cols=1241136, bs=3
    #         total: nonzeros=55301310, allocated nonzeros=55301310
    #         total number of mallocs used during MatSetValues calls=0
    #           has attached near null space
    #           using I-node (on process 0) routines: found 101223 nodes, limit used is 5
    #   Up solver (post-smoother) same as down solver (pre-smoother)
    #   linear system matrix = precond matrix:
    #   Mat Object: 4 MPI processes
    #     type: mpiaij
    #     rows=1241136, cols=1241136, bs=3
    #     total: nonzeros=55301310, allocated nonzeros=55301310
    #     total number of mallocs used during MatSetValues calls=0
    #       has attached near null space
    #       using I-node (on process 0) routines: found 101223 nodes, limit used is 5

    opts[f"{option_prefix}ksp_type"] = "cg"
    # opts[f"{option_prefix}pc_type"] = "gamg"
    # opts[f"{option_prefix}rtol"] = 1.0e-6
    # opts[f"{option_prefix}pc_gamg_coarse_eq_limit"] = 1000
    # opts[f"{option_prefix}mg_levels_ksp_type"] = "chebyshev"
    # opts[f"{option_prefix}mg_levels_pc_type"] = "jacobi"
    # opts[f"{option_prefix}mg_levels_esteig_ksp_type"] = "cg"
    # opts[f"{option_prefix}matptap_via"] = "scalable"
    opts[f"{option_prefix}pc_type"] = "gamg"
    opts[f"{option_prefix}pc_gamg_type"] = "agg"
    opts[f"{option_prefix}pc_gamg_coarse_eq_limit"] = 1000
    opts[f"{option_prefix}pc_gamg_sym_graph"] = True
    opts[f"{option_prefix}mg_levels_ksp_type"] = "chebyshev"
    opts[f"{option_prefix}mg_levels_pc_type"] = "sor"
    opts[f"{option_prefix}mg_levels_esteig_ksp_type"] = "cg"
    opts[f"{option_prefix}matptap_via"] = "scalable"
    opts[f"{option_prefix}pc_gamg_square_graph"] = 2
    opts[f"{option_prefix}pc_gamg_threshold"] = 1e-2
    opts[f"{option_prefix}help"] = None  # List all available options

    # View solver options
    # opts[f"{option_prefix}ksp_view"] = None
    ksp.setFromOptions()

    # Solve non-linear problem
    dolfinx.log.set_log_level(dolfinx.log.LogLevel.INFO)
    with dolfinx.common.Timer(f"{refinement} Solve Nitsche"):
        n, converged = solver.solve(u)
    u.x.scatter_forward()
    if solver.error_on_nonconvergence:
        assert(converged)
    it = ksp.getIterationNumber()
    rank = MPI.COMM_WORLD.rank
    if rank == 0:
        print(f"Step {load_step: d}, ndofs: {V.dofmap.index_map_bs * V.dofmap.index_map.size_global}, "
              + f"Number of Newton interations: {n: d}, Number of linear iterations: {it: d}")

    with dolfinx.io.XDMFFile(MPI.COMM_WORLD, f"results/u_nitsche_{refinement}_{load_step}.xdmf", "w") as xdmf:
        xdmf.write_mesh(mesh)
        xdmf.write_function(u)

    return u
