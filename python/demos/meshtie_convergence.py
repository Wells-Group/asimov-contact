# Copyright (C) 2022 Sarah Roggendorf
#
# SPDX-License-Identifier:    MIT

import argparse

from mpi4py import MPI
from petsc4py import PETSc

import numpy as np
import numpy.typing as npt

import ufl
from dolfinx import default_scalar_type, log
from dolfinx.common import Timer, TimingType, list_timings, timing
from dolfinx.fem import Constant, Function, assemble_scalar, dirichletbc, form, functionspace, locate_dofs_topological
from dolfinx.fem.petsc import apply_lifting, assemble_matrix, assemble_vector, create_vector, set_bc
from dolfinx.graph import adjacencylist
from dolfinx.io import XDMFFile
from dolfinx.mesh import meshtags
from dolfinx_contact.cpp import MeshTie, Problem
from dolfinx_contact.helpers import (epsilon, lame_parameters,
                                     rigid_motions_nullspace,
                                     rigid_motions_nullspace_subdomains,
                                     sigma_func)
from dolfinx_contact.meshing import (create_split_box_2D, create_split_box_3D,
                                     create_unsplit_box_2d,
                                     create_unsplit_box_3d, horizontal_sine)
from dolfinx_contact.parallel_mesh_ghosting import create_contact_mesh


# manufactured solution 2D
def u_fun_2d(x: npt.NDArray[np.float64], d: float, gdim: int) -> npt.NDArray[np.float64]:
    u2 = d * np.sin(2 * np.pi * x[0] / 5) * np.sin(2 * np.pi * x[1])
    vals = np.zeros((gdim, x.shape[1]))
    vals[1, :] = u2[:]
    return vals


# forcing 2D for manufactured solution
# this is -div(sigma(u_fun_2d))
def fun_2d(x: npt.NDArray[np.float64], d: float, mu: float, lmbda: float, gdim: int) -> npt.NDArray[np.float64]:
    a = 2 * np.pi / 5
    b = 2 * np.pi

    vals = np.zeros((gdim, x.shape[1]))
    f1 = -(lmbda + mu) * a * b * np.cos(a * x[0]) * np.cos(b * x[1])

    f2 = (mu * a**2 + (2 * mu + lmbda) * b**2) * \
        np.sin(a * x[0]) * np.sin(b * x[1])
    vals[0, :] = d * f1[:]
    vals[1, :] = d * f2[:]

    return vals

# manufacture soltuion 3D


def u_fun_3d(x: npt.NDArray[np.float64], d: float, gdim: int) -> npt.NDArray[np.float64]:
    u2 = d * np.sin(2 * np.pi * x[0] / 5) * \
        np.sin(2 * np.pi * x[1]) * np.sin(2 * np.pi * x[2])
    vals = np.zeros((gdim, x.shape[1]))
    vals[1, :] = u2[:]
    return vals


# forcing 2D for manufactured solution
# this is -div(sigma(u_fun_3d))
def fun_3d(x: npt.NDArray[np.float64], d: float, mu: float, lmbda: float, gdim: int) -> npt.NDArray[np.float64]:
    a = 2 * np.pi / 5
    b = 2 * np.pi
    c = 2 * np.pi
    f1 = -(lmbda + mu) * a * b * np.cos(a * x[0]) * np.cos(b * x[1]) * np.sin(c * x[2])

    f2 = (mu * (a**2 + c**2) + (2 * mu + lmbda) * b**2) * \
        np.sin(a * x[0]) * np.sin(b * x[1]) * np.sin(c * x[2])

    f3 = -(lmbda + mu) * b * c * np.sin(a * x[0]) * np.cos(b * x[1]) * np.cos(c * x[2])
    vals = np.zeros((gdim, x.shape[1]))
    vals[0, :] = d * f1[:]
    vals[1, :] = d * f2[:]
    vals[2, :] = d * f3[:]
    return vals


def unsplit_domain(threed: bool = False, runs: int = 1):
    '''
        This function computes the finite element solution on a conforming
        mesh that aligns with the surface that is used for splitting the domain
        in 'test_meshtie' below
        threed: tdim=gdim=3 if True, 2 otherwise
        runs: number of refinements
    '''
    # arrays to store
    errors = []
    ndofs = []
    times = []
    its = []

    res = 0.6  # mesh resolution (input to gmsh)
    # parameter for surface approximation
    num_segments = 2 * np.ceil(5.0 / (1.2 * 0.7)).astype(np.int32)

    for i in range(1, runs + 1):
        print(f"Run {i}")
        # create mesh
        if threed:
            fname = f"./meshes/box_3D_{i}"
            create_unsplit_box_3d(
                res=res, num_segments=num_segments, fname=fname)
            fun = fun_3d
            u_fun = u_fun_3d
        else:
            fname = f"./meshes/box_2D_{i}"
            create_unsplit_box_2d(
                res=res, num_segments=num_segments, filename=fname)
            fun = fun_2d
            u_fun = u_fun_2d

        # read in mesh and markers
        with XDMFFile(MPI.COMM_WORLD, f"{fname}.xdmf", "r") as xdmf:
            mesh = xdmf.read_mesh(name="Grid")
        tdim = mesh.topology.dim
        gdim = mesh.geometry.dim
        mesh.topology.create_connectivity(tdim - 1, 0)
        mesh.topology.create_connectivity(tdim - 1, tdim)
        with XDMFFile(MPI.COMM_WORLD, f"{fname}.xdmf", "r") as xdmf:
            facet_marker = xdmf.read_meshtags(mesh, name="contact_facets")

        # Compute lame parameters
        E = 1e3
        nu = 0.1
        mu_func, lambda_func = lame_parameters(False)
        mu = mu_func(E, nu)
        lmbda = lambda_func(E, nu)
        sigma = sigma_func(mu, lmbda)

        # Functions space and FEM functions
        V = functionspace(mesh, ("Lagrange", 1, (mesh.geometry.dim, )))
        ndofs.append(V.dofmap.index_map_bs * V.dofmap.index_map.size_global)
        f = Function(V)
        c = 0.01  # amplitude of solution
        f.interpolate(lambda x: fun(x, c, mu, lmbda, gdim))
        v = ufl.TestFunction(V)
        u = ufl.TrialFunction(V)

        # Boundary conditions
        facets = facet_marker.find(2)
        bc = dirichletbc(np.zeros(tdim, dtype=default_scalar_type),
                         locate_dofs_topological(V, entity_dim=tdim - 1, entities=facets), V=V)

        dx = ufl.Measure("dx", domain=mesh)
        J = form(ufl.inner(sigma(u), epsilon(v)) * dx)
        F = form(ufl.inner(f, v) * dx)

        A = assemble_matrix(J, bcs=[bc])
        A.assemble()

        # Set null-space
        null_space = rigid_motions_nullspace(V)
        A.setNearNullSpace(null_space)

        b = assemble_vector(F)
        apply_lifting(b, [J], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD,       # type: ignore
                      mode=PETSc.ScatterMode.REVERSE)  # type: ignore
        set_bc(b, [bc])

        # Set solver options
        opts = PETSc.Options()  # type: ignore
        opts["ksp_type"] = "cg"
        opts["ksp_rtol"] = 1.0e-10
        opts["pc_type"] = "gamg"

        # Use Chebyshev smoothing for multigrid
        opts["mg_levels_ksp_type"] = "chebyshev"
        opts["mg_levels_pc_type"] = "jacobi"

        # Improve estimate of eigenvalues for Chebyshev smoothing
        opts["mg_levels_ksp_chebyshev_esteig_steps"] = 20

        # Create PETSc Krylov solver and turn convergence monitoring on
        solver = PETSc.KSP().create(mesh.comm)  # type: ignore
        solver.setFromOptions()

        # Set matrix operator
        solver.setOperators(A)

        uh = Function(V)

        # Set a monitor, solve linear system, and display the solver
        # configuration
        solver.setMonitor(lambda _, its, rnorm: print(
            f"Iteration: {its}, rel. residual: {rnorm}"))
        timing_str = "~Krylov Solver"
        with Timer(timing_str):
            solver.solve(b, uh.vector)

        times.append(timing(timing_str)[1])
        its.append(solver.getIterationNumber())
        solver.destroy()
        # Scatter forward the solution vector to update ghost values
        uh.x.scatter_forward()

        # Error computation
        V_err = functionspace(mesh, ("Lagrange", 3, (mesh.geometry.dim, )))
        u_ex = Function(V_err)
        u_ex.interpolate(lambda x: u_fun(x, c, gdim))

        error_form = form(ufl.inner(u_ex - uh, u_ex - uh) * dx)
        error = assemble_scalar(error_form)
        errors.append(np.sqrt(mesh.comm.allreduce(error, op=MPI.SUM)))
        res = 0.5 * res
        num_segments = 2 * num_segments
    ncells = mesh.topology.index_map(tdim).size_local
    indices = np.array(range(ncells), dtype=np.int32)
    values = mesh.comm.rank * np.ones(ncells, dtype=np.int32)
    process_marker = meshtags(mesh, tdim, indices, values)
    process_marker.name = "process_marker"
    with XDMFFile(mesh.comm, "results/partitioning_unsplit.xdmf", "w") as xdmf:
        xdmf.write_mesh(mesh)
        xdmf.write_meshtags(process_marker, mesh.geometry)
    print("L2-error: ", errors)
    print("Number of dofs: ", ndofs)
    print("Linear solver time: ", times)
    print("Krylov iterations: ", its)


def test_meshtie(threed: bool = False, simplex: bool = True, runs: int = 5):
    '''
        This function computes the finite element solution on mesh
        split along a surface, where the mesh and the surface discretisation
        are not matching along the surface
        threed: tdim=gdim=3 if True, 2 otherwise
        simplex: If true use tet/triangle mesh if false use hex/quad mesh
        runs: number of refinements
    '''
    res = 0.8 if simplex else 1.2

    # parameter for surface approximation
    num_segments = (2 * np.ceil(5.0 / 1.2).astype(np.int32),
                    2 * np.ceil(5.0 / (1.2 * 0.7)).astype(np.int32))
    c = 0.01  # amplitude of manufactured solution

    # Nitsche parameters
    gamma = 10
    theta = 1

    # Solver options
    ksp_tol = 1e-10
    petsc_options = {
        "matptap_via": "scalable",
        "ksp_type": "cg",
        "ksp_rtol": ksp_tol,
        "pc_type": "gamg",
        "mg_levels_ksp_type": "chebyshev",
        "mg_levels_pc_type": "jacobi",
        "pc_gamg_coarse_eq_limit": 100,
        "mg_levels_ksp_chebyshev_esteig_steps": 20
    }
    errors = []
    times = []
    iterations = []
    dofs = []
    for i in range(1, runs + 1):
        print(f"Run {i}")
        if threed:
            fname = fname = f"meshes/beam3D_{i}"
            create_split_box_3D(fname, res=res, L=5.0, H=1.0, W=1.0, domain_1=[0, 1, 5, 4], domain_2=[4, 5, 2, 3], x0=[
                0, 0.5], x1=[5.0, 0.7], curve_fun=horizontal_sine, num_segments=num_segments, hex=not simplex)
            fun = fun_3d
            u_fun = u_fun_3d
        else:
            fname = fname = f"meshes/beam_{i}"
            create_split_box_2D(fname, res=res, L=5.0, H=1.0, domain_1=[0, 1, 5, 4], domain_2=[4, 5, 2, 3], x0=[
                0, 0.5], x1=[5.0, 0.7], curve_fun=horizontal_sine, num_segments=num_segments, quads=not simplex)
            fun = fun_2d
            u_fun = u_fun_2d
        with XDMFFile(MPI.COMM_WORLD, f"{fname}.xdmf", "r") as xdmf:
            mesh = xdmf.read_mesh(name="Grid")
            tdim = mesh.topology.dim
            gdim = mesh.geometry.dim
            mesh.topology.create_connectivity(tdim - 1, 0)
            mesh.topology.create_connectivity(tdim - 1, tdim)
            domain_marker = xdmf.read_meshtags(mesh, name="domain_marker")
            facet_marker = xdmf.read_meshtags(mesh, name="contact_facets")

        if mesh.comm.size > 1:
            mesh, facet_marker, domain_marker = create_contact_mesh(
                mesh, facet_marker, domain_marker, [4, 6])

        # Compute lame parameters
        E = 1e3
        nu = 0.1
        mu_func, lambda_func = lame_parameters(False)
        V2 = functionspace(mesh, ("Discontinuous Lagrange", 0))
        lmbda = Function(V2)
        lmbda_val = lambda_func(E, nu)
        lmbda.interpolate(lambda x: np.full((1, x.shape[1]), lmbda_val))
        mu = Function(V2)
        mu_val = mu_func(E, nu)
        mu.interpolate(lambda x: np.full((1, x.shape[1]), mu_val))
        sigma = sigma_func(mu, lmbda)

        # Function, TestFunction, TrialFunction and measures
        V = functionspace(mesh, ("Lagrange", 1, (mesh.geometry.dim, )))
        v = ufl.TestFunction(V)
        w = ufl.TrialFunction(V)
        dx = ufl.Measure("dx", domain=mesh, subdomain_data=domain_marker)
        ds = ufl.Measure("ds", domain=mesh, subdomain_data=facet_marker)
        h = ufl.CellDiameter(mesh)
        n = ufl.FacetNormal(mesh)

        J = ufl.inner(sigma(w), epsilon(v)) * dx

        # forcing
        def force_func(x):
            return fun(x, c, mu_val, lmbda_val, gdim)
        f = Function(V)
        f.interpolate(force_func)
        F = ufl.inner(f, v) * dx

        # 0 dirichlet
        if gdim == 3:
            g = Constant(mesh, default_scalar_type((0.0, 0.0, 0.0)))
        else:
            g = Constant(mesh, default_scalar_type((0.0, 0.0)))
        for tag in [2, 6]:
            J += - ufl.inner(sigma(w) * n, v) * ds(tag)\
                - theta * ufl.inner(sigma(v) * n, w) * \
                ds(tag) + E * gamma / h * ufl.inner(w, v) * ds(tag)
            F += - theta * ufl.inner(sigma(v) * n, g) * \
                ds(tag) + E * gamma / h * ufl.inner(g, v) * ds(tag)

        # compile forms
        cffi_options = ["-Ofast", "-march=native"]
        jit_options = {"cffi_extra_compile_args": cffi_options,
                       "cffi_libraries": ["m"]}
        F = form(F, jit_options=jit_options)
        J = form(J, jit_options=jit_options)

        # surface data for Nitsche
        contact = [(0, 2), (0, 3), (1, 2), (1, 3),
                   (2, 0), (2, 1), (3, 0), (3, 1)]
        data = np.array([3, 4, 7, 8], dtype=np.int32)
        offsets = np.array([0, 4], dtype=np.int32)
        surfaces = adjacencylist(data, offsets)

        # initialise meshties
        meshties = MeshTie([facet_marker._cpp_object], surfaces, contact,
                           mesh._cpp_object, quadrature_degree=5)
        meshties.generate_kernel_data(Problem.Elasticity, V._cpp_object, {
                                      "lambda": lmbda._cpp_object, "mu": mu._cpp_object}, E * gamma, theta)

        # create matrix, vector
        A = meshties.create_matrix(J._cpp_object)
        b = create_vector(F)

        # Assemble right hand side
        b.zeroEntries()
        b.ghostUpdate(addv=PETSc.InsertMode.INSERT,    # type: ignore
                      mode=PETSc.ScatterMode.FORWARD)  # type: ignore
        assemble_vector(b, F)
        b.ghostUpdate(addv=PETSc.InsertMode.ADD,       # type: ignore
                      mode=PETSc.ScatterMode.REVERSE)  # type: ignore

        # Assemble matrix
        A.zeroEntries()
        meshties.assemble_matrix(A, V._cpp_object, Problem.Elasticity)
        assemble_matrix(A, J)
        A.assemble()

        # Set rigid motion nullspace
        null_space = rigid_motions_nullspace_subdomains(V, domain_marker, np.unique(domain_marker.values),
                                                        num_domains=2)
        A.setNearNullSpace(null_space)

        # Create PETSc Krylov solver and turn convergence monitoring on
        opts = PETSc.Options()  # type: ignore
        for key in petsc_options:
            opts[key] = petsc_options[key]
        solver = PETSc.KSP().create(mesh.comm)  # type: ignore
        solver.setFromOptions()

        # Set matrix operator
        solver.setOperators(A)

        u1 = Function(V)

        dofs_global = V.dofmap.index_map_bs * V.dofmap.index_map.size_global
        log.set_log_level(log.LogLevel.OFF)
        # Set a monitor, solve linear system, and display the solver
        # configuration
        solver.setMonitor(lambda _, its, rnorm: print(
            f"Iteration: {its}, rel. residual: {rnorm}"))
        timing_str = "~Contact : Krylov Solver"
        with Timer(timing_str):
            solver.solve(b, u1.vector)

        # Scatter forward the solution vector to update ghost values
        u1.x.scatter_forward()
        solver_time = timing(timing_str)[1]

        V_err = functionspace(mesh, ("Lagrange", 3, (mesh.geometry.dim, )))
        u_ex = Function(V_err)
        u_ex.interpolate(lambda x: u_fun(x, c, gdim))

        dx = ufl.Measure("dx", domain=mesh)
        error_form = form(ufl.inner(u_ex - u1, u_ex - u1) * dx)
        error = assemble_scalar(error_form)
        errors.append(np.sqrt(mesh.comm.allreduce(error, op=MPI.SUM)))
        res = 0.5 * res
        num_segments = (2 * num_segments[0], 2 * num_segments[1])
        iterations.append(solver.getIterationNumber())
        times.append(solver_time)
        dofs.append(dofs_global)

    ncells = mesh.topology.index_map(tdim).size_local
    indices = np.array(range(ncells), dtype=np.int32)
    values = mesh.comm.rank * np.ones(ncells, dtype=np.int32)
    process_marker = meshtags(mesh, tdim, indices, values)
    process_marker.name = "process_marker"
    with XDMFFile(mesh.comm, "results/partitioning_split.xdmf", "w") as xdmf:
        xdmf.write_mesh(mesh)
        xdmf.write_meshtags(process_marker, mesh.geometry)
    list_timings(mesh.comm, [TimingType.wall])
    print("L2 errors; ", errors)
    print("Solver time: ", times)
    print("Krylov iterations: ", iterations)
    print("Number of dofs: ", dofs)


if __name__ == "__main__":
    desc = "Meshtie"
    parser = argparse.ArgumentParser(description=desc,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--runs", default=1, type=int, dest="runs",
                        help="Number of refinements")
    _3D = parser.add_mutually_exclusive_group(required=False)
    _3D.add_argument('--3D', dest='threed', action='store_true',
                     help="Use 3D mesh", default=False)
    _simplex = parser.add_mutually_exclusive_group(required=False)
    _simplex.add_argument('--simplex', dest='simplex', action='store_true',
                          help="Use triangle/tet mesh", default=False)
    _unsplit = parser.add_mutually_exclusive_group(required=False)
    _unsplit.add_argument('--unsplit', dest='unsplit', action='store_true',
                          help="Use conforming mesh", default=False)
    args = parser.parse_args()
    if args.unsplit:
        unsplit_domain(threed=args.threed, runs=args.runs)
    else:
        test_meshtie(simplex=args.simplex, threed=args.threed, runs=args.runs)
