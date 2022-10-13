# Copyright (C) 2022 Sarah Roggendorf
#
# SPDX-License-Identifier:    MIT

import argparse
import dolfinx.fem as _fem
from dolfinx.common import Timer, timing, TimingType, list_timings
from dolfinx.cpp.mesh import MeshTags_int32
from dolfinx.fem import Constant, Function, VectorFunctionSpace
from dolfinx.graph import create_adjacencylist
from dolfinx.io import XDMFFile
import numpy as np
import numpy.typing as npt
import ufl
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx_contact.helpers import (epsilon, lame_parameters,
                                     rigid_motions_nullspace, sigma_func)
from dolfinx_contact.meshing import (create_split_box_2D, create_split_box_3D,
                                     horizontal_sine, create_unsplit_box_2d, create_unsplit_box_3d)
from dolfinx_contact.meshtie import nitsche_meshtie
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

    f2 = (mu * a**2 + (2 * mu + lmbda) * b**2) * np.sin(a * x[0]) * np.sin(b * x[1])
    vals[0, :] = d * f1[:]
    vals[1, :] = d * f2[:]

    return vals

# manufacture soltuion 3D


def u_fun_3d(x: npt.NDArray[np.float64], d: float, gdim: int) -> npt.NDArray[np.float64]:
    u2 = d * np.sin(2 * np.pi * x[0] / 5) * np.sin(2 * np.pi * x[1]) * np.sin(2 * np.pi * x[2])
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

    f2 = (mu * (a**2 + c**2) + (2 * mu + lmbda) * b**2) * np.sin(a * x[0]) * np.sin(b * x[1]) * np.sin(c * x[2])

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
    num_segments = 2 * np.ceil(5.0 / (1.2 * 0.7)).astype(np.int32)  # parameter for surface approximation

    for i in range(1, runs + 1):
        print(f"Run {i}")
        # create mesh
        if threed:
            fname = f"box_3D_{i}"
            # create_unsplit_box_3d(res=res, num_segments=num_segments)
            fun = fun_3d
            u_fun = u_fun_3d
        else:
            fname = f"box_2D_{i}"
            # create_unsplit_box_2d(res=res, num_segments=num_segments)
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
        V = _fem.VectorFunctionSpace(mesh, ("CG", 1))
        ndofs.append(V.dofmap.index_map_bs * V.dofmap.index_map.size_global)
        f = _fem.Function(V)
        c = 0.01  # amplitude of solution
        f.interpolate(lambda x: fun(x, c, mu, lmbda, gdim))
        v = ufl.TestFunction(V)
        u = ufl.TrialFunction(V)

        # Boundary conditions
        facets = facet_marker.find(2)
        bc = _fem.dirichletbc(np.zeros(tdim, dtype=PETSc.ScalarType),
                              _fem.locate_dofs_topological(V, entity_dim=tdim - 1, entities=facets), V=V)

        dx = ufl.Measure("dx", domain=mesh)
        J = _fem.form(ufl.inner(sigma(u), epsilon(v)) * dx)
        F = _fem.form(ufl.inner(f, v) * dx)

        A = _fem.petsc.assemble_matrix(J, bcs=[bc])
        A.assemble()

        # Set null-space
        null_space = rigid_motions_nullspace(V)
        A.setNearNullSpace(null_space)

        b = _fem.petsc.assemble_vector(F)
        _fem.petsc.apply_lifting(b, [J], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        _fem.petsc.set_bc(b, [bc])

        # Set solver options
        opts = PETSc.Options()
        opts["ksp_type"] = "cg"
        opts["ksp_rtol"] = 1.0e-10
        opts["pc_type"] = "gamg"

        # Use Chebyshev smoothing for multigrid
        opts["mg_levels_ksp_type"] = "chebyshev"
        opts["mg_levels_pc_type"] = "jacobi"

        # Improve estimate of eigenvalues for Chebyshev smoothing
        opts["mg_levels_esteig_ksp_type"] = "cg"
        opts["mg_levels_ksp_chebyshev_esteig_steps"] = 20

        # Create PETSc Krylov solver and turn convergence monitoring on
        solver = PETSc.KSP().create(mesh.comm)
        solver.setFromOptions()

        # Set matrix operator
        solver.setOperators(A)

        uh = _fem.Function(V)

        # Set a monitor, solve linear system, and display the solver
        # configuration
        solver.setMonitor(lambda _, its, rnorm: print(f"Iteration: {its}, rel. residual: {rnorm}"))
        timing_str = "~Krylov Solver"
        with Timer(timing_str):
            solver.solve(b, uh.vector)

        times.append(timing(timing_str)[1])
        its.append(solver.getIterationNumber())

        # Scatter forward the solution vector to update ghost values
        uh.x.scatter_forward()

        # Error computation
        V_err = _fem.VectorFunctionSpace(mesh, ("CG", 3))
        u_ex = _fem.Function(V_err)
        u_ex.interpolate(lambda x: u_fun(x, c, gdim))

        error_form = _fem.form(ufl.inner(u_ex - uh, u_ex - uh) * dx)
        error = _fem.assemble_scalar(error_form)
        errors.append(np.sqrt(mesh.comm.allreduce(error, op=MPI.SUM)))
        res = 0.5 * res
        num_segments = 2 * num_segments
    ncells = mesh.topology.index_map(tdim).size_local
    indices = np.array(range(ncells), dtype=np.int32)
    values = mesh.comm.rank * np.ones(ncells, dtype=np.int32)
    process_marker = MeshTags_int32(mesh, tdim, indices, values)
    process_marker.name = "process_marker"
    with XDMFFile(mesh.comm, "results/partitioning_unsplit.xdmf", "w") as xdmf:
        xdmf.write_mesh(mesh)
        xdmf.write_meshtags(process_marker)
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
    num_segments = (2 * np.ceil(5.0 / 1.2).astype(np.int32), 2 * np.ceil(5.0 / (1.2 * 0.7)).astype(np.int32))
    c = 0.01  # amplitude of manufactured solution
    # Compute lame parameters
    E = 1e3
    nu = 0.1
    mu_func, lambda_func = lame_parameters(False)
    mu = mu_func(E, nu)
    lmbda = lambda_func(E, nu)
    sigma = sigma_func(mu, lmbda)

    # dictionary with problem parameters
    gamma = 10
    theta = 1
    problem_parameters = {"mu": mu, "lambda": lmbda, "gamma": E * gamma, "theta": theta}

    # Solver options
    ksp_tol = 1e-10
    petsc_options = {
        "matptap_via": "scalable",
        "ksp_type": "cg",
        "ksp_rtol": ksp_tol,
        "pc_type": "gamg",
        "mg_levels_ksp_type": "chebyshev",
        "mg_levels_pc_type": "jacobi",
        "mg_levels_esteig_ksp_type": "cg",
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
            fname = fname = f"beam3D_{i}"
            # create_split_box_3D(fname, res=res, L=5.0, H=1.0, W=1.0, domain_1=[0, 1, 5, 4], domain_2=[4, 5, 2, 3], x0=[
            #     0, 0.5], x1=[5.0, 0.7], curve_fun=horizontal_sine, num_segments=num_segments, hex=not simplex)
            fun = fun_3d
            u_fun = u_fun_3d
        else:
            fname = fname = f"beam_{i}"
            # create_split_box_2D(fname, res=res, L=5.0, H=1.0, domain_1=[0, 1, 5, 4], domain_2=[4, 5, 2, 3], x0=[
            #     0, 0.5], x1=[5.0, 0.7], curve_fun=horizontal_sine, num_segments=num_segments, quads=not simplex)
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

        mesh, facet_marker, domain_marker = create_contact_mesh(
            mesh, facet_marker, domain_marker, [4, 6])

        # Function, TestFunction, TrialFunction and measures
        V = VectorFunctionSpace(mesh, ("CG", 1))
        u = Function(V)
        v = ufl.TestFunction(V)
        w = ufl.TrialFunction(V)
        dx = ufl.Measure("dx", domain=mesh, subdomain_data=domain_marker)
        ds = ufl.Measure("ds", domain=mesh, subdomain_data=facet_marker)
        h = ufl.CellDiameter(mesh)
        n = ufl.FacetNormal(mesh)

        J = ufl.inner(sigma(w), epsilon(v)) * dx

        # forcing
        def force_func(x):
            return fun(x, c, mu, lmbda, gdim)
        f = Function(V)
        f.interpolate(force_func)
        F = ufl.inner(f, v) * dx

        # 0 dirichlet
        if gdim == 3:
            g = Constant(mesh, PETSc.ScalarType((0.0, 0.0, 0.0)))
        else:
            g = Constant(mesh, PETSc.ScalarType((0.0, 0.0)))
        for tag in [3, 5]:
            J += - ufl.inner(sigma(w) * n, v) * ds(tag)\
                - theta * ufl.inner(sigma(v) * n, w) * \
                ds(tag) + E * gamma / h * ufl.inner(w, v) * ds(tag)
            F += - theta * ufl.inner(sigma(v) * n, g) * \
                ds(tag) + E * gamma / h * ufl.inner(g, v) * ds(tag)

        contact = [(1, 0), (0, 1)]
        data = np.array([4, 6], dtype=np.int32)
        offsets = np.array([0, 2], dtype=np.int32)
        surfaces = create_adjacencylist(data, offsets)
        # Solve contact problem using Nitsche's method
        u1, its, solver_time, ndofs = nitsche_meshtie(lhs=J, rhs=F, u=u, markers=[domain_marker, facet_marker],
                                                      surface_data=(surfaces, contact),
                                                      bcs=[], problem_parameters=problem_parameters,
                                                      petsc_options=petsc_options)

        u1.x.scatter_forward()

        V_err = _fem.VectorFunctionSpace(mesh, ("CG", 3))
        u_ex = _fem.Function(V_err)
        u_ex.interpolate(lambda x: u_fun(x, c, gdim))

        dx = ufl.Measure("dx", domain=mesh)
        error_form = _fem.form(ufl.inner(u_ex - u1, u_ex - u1) * dx)
        error = _fem.assemble_scalar(error_form)
        errors.append(np.sqrt(mesh.comm.allreduce(error, op=MPI.SUM)))
        res = 0.5 * res
        num_segments = (2 * num_segments[0], 2 * num_segments[1])
        iterations.append(its)
        times.append(solver_time)
        dofs.append(ndofs)

    ncells = mesh.topology.index_map(tdim).size_local
    indices = np.array(range(ncells), dtype=np.int32)
    values = mesh.comm.rank * np.ones(ncells, dtype=np.int32)
    process_marker = MeshTags_int32(mesh, tdim, indices, values)
    process_marker.name = "process_marker"
    with XDMFFile(mesh.comm, "results/partitioning_split.xdmf", "w") as xdmf:
        xdmf.write_mesh(mesh)
        xdmf.write_meshtags(process_marker)
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
