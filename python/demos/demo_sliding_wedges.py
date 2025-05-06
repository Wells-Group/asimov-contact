# Copyright (C) 2023 Sarah Roggendorf
#
# SPDX-License-Identifier:    MIT

import argparse
import tempfile
from pathlib import Path

from mpi4py import MPI
from petsc4py.PETSc import InsertMode, ScatterMode  # type: ignore

import numpy as np
import ufl
from dolfinx import default_scalar_type
from dolfinx.fem import (
    Constant,
    Function,
    assemble_scalar,
    dirichletbc,
    form,
    functionspace,
    locate_dofs_topological,
)
from dolfinx.fem.petsc import (
    apply_lifting,
    assemble_matrix,
    assemble_vector,
    create_vector,
    set_bc,
)
from dolfinx.graph import adjacencylist
from dolfinx.io import VTXWriter, XDMFFile
from dolfinx_contact.cpp import ContactMode
from dolfinx_contact.general_contact.contact_problem import ContactProblem, FrictionLaw
from dolfinx_contact.helpers import (
    epsilon,
    lame_parameters,
    rigid_motions_nullspace_subdomains,
    sigma_func,
)
from dolfinx_contact.meshing import convert_mesh_new, sliding_wedges
from dolfinx_contact.newton_solver import NewtonSolver

if __name__ == "__main__":
    desc = "Friction example with two elastic cylinders for verifying correctness of code"
    parser = argparse.ArgumentParser(
        description=desc, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--quadrature",
        default=5,
        type=int,
        dest="q_degree",
        help="Quadrature degree used for contact integrals",
    )
    parser.add_argument(
        "--order",
        default=1,
        type=int,
        dest="order",
        help="Order of mesh geometry",
        choices=[1, 2],
    )
    _simplex = parser.add_mutually_exclusive_group(required=False)
    _simplex.add_argument(
        "--simplex",
        dest="simplex",
        action="store_true",
        help="Use triangle/tet mesh",
        default=False,
    )
    parser.add_argument("--res", default=0.1, type=np.float64, dest="res", help="Mesh resolution")

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
    fric = 0.5
    angle = np.arctan(0.1)

    # Create mesh
    outname = "results/sliding_wedges_simplex" if simplex else "results/sliding_wedges_quads"
    with tempfile.TemporaryDirectory() as tmpdirname:
        fname = Path(tmpdirname, "sliding_wedges.msh")
        sliding_wedges(fname, not simplex, args.res, args.order, angle=angle)
        convert_mesh_new(fname, fname.with_suffix(".xdmf"), gdim=2)
        with XDMFFile(MPI.COMM_WORLD, fname.with_suffix(".xdmf"), "r") as xdmf:
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

    V = functionspace(mesh, ("CG", args.order, (mesh.geometry.dim,)))
    # boundary conditions
    t = Constant(mesh, default_scalar_type((0.3, 0.0)))

    dirichlet_dofs_1 = locate_dofs_topological(V.sub(1), 1, facet_marker.find(dirichlet_bdy_1))
    dirichlet_dofs_2 = locate_dofs_topological(V, 1, facet_marker.find(dirichlet_bdy_2))

    g0 = Constant(mesh, default_scalar_type(0.0))
    g1 = Constant(mesh, default_scalar_type((0, 0)))
    bcs = [
        dirichletbc(g0, dirichlet_dofs_1, V.sub(1)),
        dirichletbc(g1, dirichlet_dofs_2, V),
    ]
    bc_fns = [g0, g1]

    # DG-0 funciton for material
    V0 = functionspace(mesh, ("DG", 0))
    mu_dg = Function(V0)
    lmbda_dg = Function(V0)
    fric_dg = Function(V0)
    mu_dg.interpolate(lambda x: np.full((1, x.shape[1]), mu))
    lmbda_dg.interpolate(lambda x: np.full((1, x.shape[1]), lmbda))
    sigma = sigma_func(mu, lmbda)
    fric_dg.interpolate(lambda x: np.full((1, x.shape[1]), fric))

    # Pack mesh data for Nitsche solver
    contact = [(0, 1), (1, 0)]
    data = np.array([contact_bdy_1, contact_bdy_2], dtype=np.int32)
    offsets = np.array([0, 2], dtype=np.int32)
    surfaces = adjacencylist(data, offsets)

    # Function, TestFunction, TrialFunction and measures
    u = Function(V)
    du = Function(V)
    w = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    dx = ufl.Measure("dx", domain=mesh, subdomain_data=domain_marker)
    ds = ufl.Measure("ds", domain=mesh, subdomain_data=facet_marker)

    # Create variational form without contact contributions
    F = ufl.inner(sigma(u), epsilon(v)) * dx

    # body forces
    F -= ufl.inner(t, v) * ds(neumann_bdy)

    F = ufl.replace(F, {u: u + du})

    J = ufl.derivative(F, du, w)

    # compiler options to improve performance
    jit_options = {"cffi_extra_compile_args": [], "cffi_libraries": ["m"]}
    # compiled forms for rhs and tangen system
    F_compiled = form(F, jit_options=jit_options)
    J_compiled = form(J, jit_options=jit_options)

    search_mode = [ContactMode.ClosestPoint, ContactMode.ClosestPoint]

    # Solver options
    ksp_tol = 1e-10
    newton_tol = 1e-7
    newton_options = {
        "relaxation_parameter": 1.0,
        "atol": newton_tol,
        "rtol": newton_tol,
        "convergence_criterion": "residual",
        "max_it": 50,
        "error_on_nonconvergence": True,
    }

    petsc_options = {
        "matptap_via": "scalable",
        "ksp_type": "gmres",
        "ksp_rtol": ksp_tol,
        "ksp_atol": ksp_tol,
        "pc_type": "gamg",
        "pc_mg_levels": 3,
        "pc_mg_cycles": 1,  # 1 is v, 2 is w
        "mg_levels_ksp_type": "chebyshev",
        "mg_levels_pc_type": "jacobi",
        "pc_gamg_type": "agg",
        "pc_gamg_coarse_eq_limit": 100,
        "pc_gamg_agg_nsmooths": 1,
        "pc_gamg_threshold": 1e-3,
        "pc_gamg_square_graph": 2,
        "pc_gamg_reuse_interpolation": False,
        "ksp_initial_guess_nonzero": False,
        "ksp_norm_type": "unpreconditioned",
    }

    def _pressure(x):
        vals = np.zeros(x.shape[1])
        return vals

    # Solve contact problem using Nitsche's method
    contact_problem = ContactProblem(
        [facet_marker], surfaces, contact, mesh, args.q_degree, search_mode
    )
    contact_problem.generate_contact_data(
        FrictionLaw.Coulomb,
        V,
        {"u": u, "du": du, "mu": mu_dg, "lambda": lmbda_dg, "fric": fric_dg},
        E * 10,
        -1,
    )

    # define functions for newton solver
    def compute_coefficients(x, coeffs):
        du.x.scatter_forward()
        contact_problem.update_contact_data(du)

    def compute_residual(x, b, coeffs):
        b.zeroEntries()
        b.ghostUpdate(addv=InsertMode.INSERT, mode=ScatterMode.FORWARD)
        contact_problem.assemble_vector(b, V)
        assemble_vector(b, F_compiled)

        # Apply boundary condition
        if len(bcs) > 0:
            apply_lifting(b, [J_compiled], bcs=[bcs], x0=[x], alpha=-1.0)
        b.ghostUpdate(addv=InsertMode.ADD, mode=ScatterMode.REVERSE)
        if len(bcs) > 0:
            set_bc(b, bcs, x, -1.0)

    def compute_jacobian_matrix(x, a_mat, coeffs):
        a_mat.zeroEntries()
        contact_problem.assemble_matrix(a_mat, V)
        assemble_matrix(a_mat, J_compiled, bcs=bcs)
        a_mat.assemble()

    # create vector and matrix
    a_mat = contact_problem.create_matrix(J_compiled)
    b = create_vector(F_compiled)

    # Set up snes solver for nonlinear solver
    newton_solver = NewtonSolver(mesh.comm, a_mat, b, contact_problem.coeffs)
    # Set matrix-vector computations
    newton_solver.set_residual(compute_residual)
    newton_solver.set_jacobian(compute_jacobian_matrix)
    newton_solver.set_coefficients(compute_coefficients)

    # Set rigid motion nullspace
    null_space = rigid_motions_nullspace_subdomains(
        V, domain_marker, np.unique(domain_marker.values), num_domains=2
    )
    newton_solver.A.setNearNullSpace(null_space)

    # Set Newton solver options
    newton_solver.set_newton_options(newton_options)

    # Set Krylov solver options
    newton_solver.set_krylov_options(petsc_options)
    # initialise vtx writer
    u.name = "u"
    vtx = VTXWriter(mesh.comm, "results/sliding_wedges.bp", [u], "bp4")
    vtx.write(0)
    n, converged = newton_solver.solve(du, write_solution=True)
    du.x.scatter_forward()
    u.x.array[:] += du.x.array[:]
    vtx.write(1)

    n = ufl.FacetNormal(mesh)
    metadata = {"quadrature_degree": 2}

    ds = ufl.Measure("ds", domain=mesh, metadata=metadata, subdomain_data=facet_marker)
    ex = Constant(mesh, default_scalar_type((1.0, 0.0)))
    ey = Constant(mesh, default_scalar_type((0.0, 1.0)))
    Rx_1form = form(ufl.inner(sigma(u) * n, ex) * ds(contact_bdy_1))
    Ry_1form = form(ufl.inner(sigma(u) * n, ey) * ds(contact_bdy_1))
    Rx_2form = form(ufl.inner(sigma(u) * n, ex) * ds(contact_bdy_2))
    Ry_2form = form(ufl.inner(sigma(u) * n, ey) * ds(contact_bdy_2))
    R_x1 = mesh.comm.allreduce(assemble_scalar(Rx_1form), op=MPI.SUM)
    R_y1 = mesh.comm.allreduce(assemble_scalar(Ry_1form), op=MPI.SUM)
    R_x2 = mesh.comm.allreduce(assemble_scalar(Rx_2form), op=MPI.SUM)
    R_y2 = mesh.comm.allreduce(assemble_scalar(Ry_2form), op=MPI.SUM)

    print(
        "Rx/Ry",
        abs(R_x1) / abs(R_y1),
        abs(R_x2) / abs(R_y2),
        (fric + np.tan(angle)) / (1 - fric * np.tan(angle)),
    )
