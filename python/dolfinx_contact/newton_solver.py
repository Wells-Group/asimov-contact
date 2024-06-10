# Copyright (C) 2022 Jørgen S. Dokken
#
# SPDX-License-Identifier:    MIT

from enum import Enum
from typing import Any, Callable, List, Optional, Tuple, Union

from mpi4py import MPI
from petsc4py import PETSc

import numpy
import numpy.typing as npt
from dolfinx import common, default_scalar_type, fem
from dolfinx.io import VTXWriter

__all__ = ["NewtonSolver", "ConvergenceCriterion"]


class ConvergenceCriterion(Enum):
    residual = 10
    incremental = 20


class NewtonSolver:
    __slots__ = [
        "max_it",
        "rtol",
        "atol",
        "report",
        "error_on_nonconvergence",
        "convergence_criterion",
        "relaxation_parameter",
        "_compute_residual",
        "_compute_jacobian",
        "_compute_preconditioner",
        "_compute_coefficients",
        "krylov_iterations",
        "iteration",
        "residual",
        "initial_residual",
        "krylov_solver",
        "_dx",
        "comm",
        "_A",
        "_b",
        "_coeffs",
        "_P",
    ]

    def __init__(
        self,
        comm: MPI.Comm,
        J: PETSc.Mat,  # type: ignore
        b: PETSc.Vec,  # type: ignore
        coeffs: List[npt.NDArray[default_scalar_type]],
    ):
        """
        Create a Newton solver

        Args:
            comm: The MPI communicator
            J: The matrix to assemble the Jacobian into
            b: The vector to assemble the residual into
        """

        self.max_it = 50
        self.rtol = 1e-7
        self.atol = 1e-7
        self.iteration = 0
        self.krylov_iterations = 0
        self.initial_residual = 0
        self.residual = 0
        self.comm = comm
        self.convergence_criterion = ConvergenceCriterion.residual
        self._A = J
        self._b = b
        self._coeffs = coeffs
        self._dx: Optional[PETSc.Vec] = None  # type: ignore
        self.krylov_solver = PETSc.KSP()  # type: ignore
        self.krylov_solver.create(self.comm)
        self.krylov_solver.setOptionsPrefix("Newton_solver_")
        self.error_on_nonconvergence = False

    def set_krylov_options(self, options: dict[str, Any]):
        """
        Set options for Krylov solver
        """
        # Options that has to apply to all matrices, not just the solver matrix
        keys = ["matptap_via"]
        g_opts = {}
        pc_keys = ["pc_mg_levels", "pc_mg_cycles"]
        pc_opts = {}
        opts = PETSc.Options()  # type: ignore
        opts.prefixPush(self.krylov_solver.getOptionsPrefix())
        for k, v in options.items():
            if k in keys:
                g_opts[k] = v
            elif k in pc_keys:
                pc_opts[k] = v
            else:
                opts[k] = v
        opts.prefixPop()
        for k, v in g_opts.items():
            opts[k] = v
        self.krylov_solver.setFromOptions()
        pc = self.krylov_solver.getPC()
        if pc_opts.get("pc_mg_levels") is not None:
            pc.setMGLevels(pc_opts.get("pc_mg_levels"))
            if pc_opts.get("pc_mg_cycles") is not None:
                pc.setMGCycleType(pc_opts.get("pc_mg_cycles"))
        self._A.setOptionsPrefix(self.krylov_solver.getOptionsPrefix())
        self._A.setFromOptions()
        self._b.setOptionsPrefix(self.krylov_solver.getOptionsPrefix())
        self._b.setFromOptions()

    def update_krylov_solver(self, options: dict[str, Any]):
        """
        Create new krylov solver with different set of options
        """
        self.krylov_solver = PETSc.KSP()  # type: ignore
        self.krylov_solver.create(self.comm)
        self.krylov_solver.setOptionsPrefix("Newton_solver_")
        self.set_krylov_options(options)

    def solve(
        self,
        u: Union[fem.Function, PETSc.Vec],  # type: ignore
        write_solution: bool = False,
        offset_fun: Union[fem.Function, None] = None,
    ):
        """
        Solve non-linear problem into function u.
        Returns the number of iterations and if the solver converged
        """
        try:
            n, converged = self._solve(u, write_solution, offset_fun)
            u.x.scatter_forward()
        except AttributeError:
            write_solution = False
            n, converged = self._solve(u)
            u.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)  # type: ignore
        return n, converged

    @property
    def A(self) -> PETSc.Mat:  # type: ignore
        """Get the Jacobian matrix"""
        return self._A

    @property
    def b(self) -> PETSc.Vec:  # type: ignore
        """Get the residual vector"""
        return self._b

    def set_jacobian(
        self,
        func: Callable[  # type: ignore
            [PETSc.Vec, PETSc.Mat, List[npt.NDArray[default_scalar_type]]], None
        ],
    ):
        """
        Set the function for computing the Jacobian
        Args:
            func: Function to compute the Jacobian matrix.
        """
        self._compute_jacobian = func

    def set_petsc_matrix(self, a_mat: PETSc.Mat):  # type: ignore
        self._A = a_mat

    def set_residual(
        self,
        func: Callable[  # type: ignore
            [PETSc.Vec, PETSc.Vec, List[npt.NDArray[default_scalar_type]]], None
        ],
    ):
        """
        Set the function for computing the residual
        Args:
            func: Function to compute the residual
        """
        self._compute_residual = func

    def set_preconditioner(
        self,
        func: Callable[  # type: ignore
            [PETSc.Vec, PETSc.Mat, List[npt.NDArray[default_scalar_type]]], None
        ],
        P: PETSc.Mat,  # type: ignore
    ):
        """
        Set the function for computing the preconditioner matrix
        Args:
            func: Function to compute the preconditioner matrix b (x, P)
            P: The matrix to assemble the preconditioner into
        """
        self._compute_preconditioner = func
        self._P = P

    def set_coefficients(self, func: Callable[[PETSc.Vec, List[npt.NDArray[default_scalar_type]]], None]):  # type: ignore
        """
        Set the function for computing the coefficients needed for assembly
        Args:
            func: Function to compute coefficients coeffs(x)
        """
        self._compute_coefficients = func

    def set_newton_options(self, options: dict):
        """
        Set Newton options from a dictionary
        """
        atol = options.get("atol")
        if atol is not None:
            self.atol = atol

        rtol = options.get("rtol")
        if rtol is not None:
            self.rtol = rtol

        crit = options.get("convergence_criterion")
        if crit is not None:
            if crit == "residual":
                self.convergence_criterion = ConvergenceCriterion.residual
            elif crit == "incremental":
                self.convergence_criterion = ConvergenceCriterion.incremental
            else:
                raise RuntimeError("Unknown Convergence criterion")
        max_it = options.get("max_it")
        if max_it is not None:
            self.max_it = max_it
        e_convg = options.get("error_on_nonconvergence")
        if e_convg is not None:
            self.error_on_nonconvergence = e_convg
        relax = options.get("relaxation_parameter")
        if relax is not None:
            self.relaxation_parameter = relax
        else:
            self.relaxation_parameter = 1.0

    def _pre_computation(self, x: PETSc.Vec):  # type: ignore
        x.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)  # type: ignore

    def _post_solve(self, x: PETSc.Vec):  # type: ignore
        x.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)  # type: ignore

    def _check_convergence(self, r: PETSc.Vec):  # type: ignore
        residual = r.norm(PETSc.NormType.NORM_2)  # type: ignore
        try:
            relative_residual = residual / self.initial_residual
        except ZeroDivisionError:
            relative_residual = numpy.inf
        if self.comm.rank == 0:
            print(
                f"Newton Iteration {self.iteration}: r (abs) {residual} (atol={self.atol})",
                flush=True,
                end=" ",
            )
            print(f"r (rel) {relative_residual} (rtol={self.rtol})", flush=True, end=" ")
            # Petsc KSP converged reason:
            # https://petsc.org/main/docs/manualpages/KSP/KSPConvergedReason/
            print(
                f"Krylov iterations: {self.krylov_solver.getIterationNumber()}",
                flush=True,
                end=" ",
            )
            print(
                f"converged: {self.krylov_solver.getConvergedReason()}",
                flush=True,
                end=" ",
            )
            print(f"relaxation: {self.relaxation_parameter}")
        return residual, relative_residual < self.rtol and residual < self.atol

    def _update_solution(self, dx: PETSc.Vec, x: PETSc.Vec):  # type: ignore
        """
        Compute x-= relaxation_patameter*dx
        """
        x.axpy(-self.relaxation_parameter, dx)

    def _solve(
        self,
        x: Union[PETSc.Vec, fem.Function],  # type: ignore
        write_solution: bool = False,
        u: Union[fem.Function, None] = None,
    ) -> Tuple[int, int]:
        t = common.Timer("~Contact: Newton (Newton solver)")
        try:
            x_vec = x.petsc_vec
        except AttributeError:
            x_vec = x
            write_solution = False

        if write_solution:
            V = x.function_space
            mesh = x.function_space.mesh
            if u is not None:
                out = fem.Function(V)
                out.x.array[:] = x.x.array[:] + u.x.array[:]
            else:
                out = x
            vtx = VTXWriter(mesh.comm, "results/newton.bp", [out])
            vtx.write(0)
        # Reset iteration counts
        self.iteration = 0
        self.krylov_iterations = 0
        self.residual = -1
        rel_min = 1e-2
        res_old = numpy.inf

        try:
            self._compute_coefficients(x_vec, self._coeffs)
        except AttributeError:
            raise RuntimeError("Function for computing coefficients has not been set")

        try:
            self._pre_computation(x_vec)
        except AttributeError:
            raise RuntimeError("Pre-computation has not been set")

        try:
            self._compute_residual(x_vec, self._b, self._coeffs)
        except AttributeError:
            raise RuntimeError("Function for computing residual vector has not been provided")

        newton_converged = False
        res_old = self._b.norm(PETSc.NormType.NORM_2)  # type: ignore
        if self.convergence_criterion == ConvergenceCriterion.residual:
            self.residual, newton_converged = self._check_convergence(self.b)
        elif self.convergence_criterion == ConvergenceCriterion.incremental:
            # We need to do at least one Newton step with the ||dx||-stopping criterion
            newton_converged = False
        else:
            raise ValueError("Unknown convergence criterion")

        try:
            self.krylov_solver.setOperators(self._A, self._P)
        except AttributeError:
            self.krylov_solver.setOperators(self._A, self._A)

        if self._dx is None:
            self._dx = self._A.createVecRight()

        x_copy = self._A.createVecRight()

        # Start iterations
        while not newton_converged and self.iteration < self.max_it:
            x_copy.array_w[:] = x_vec.array_r[:]
            self.relaxation_parameter = 1.0
            try:
                self._compute_jacobian(x_vec, self._A, self._coeffs)
            except AttributeError:
                raise RuntimeError("Function for computing Jacobian has not been provided")

            try:
                self._compute_preconditioner(x_vec, self._P, self._coeffs)
            except AttributeError:
                pass

            # Perform linear solve and update number of Krylov iterations
            with common.Timer("~Contact: Newton (Krylov solver)"):
                self.krylov_solver.solve(self._b, self._dx)
            self.krylov_iterations += self.krylov_solver.getIterationNumber()
            assert self.krylov_solver.getConvergedReason() > 0

            # Update solution
            self._update_solution(self._dx, x_vec)

            self._compute_coefficients(x_vec, self._coeffs)

            # Increment iteration count
            self.iteration += 1

            # Update internal variables prior to computing residual
            try:
                self._post_solve(x_vec)
            except AttributeError:
                pass

            # Compute residual (F)
            self._compute_residual(x_vec, self._b, self._coeffs)

            res_new = self._b.norm(PETSc.NormType.NORM_2)  # type: ignore

            # adaptive relaxation
            while res_old < res_new:
                self.relaxation_parameter = 0.5 * self.relaxation_parameter
                if self.relaxation_parameter < rel_min:
                    self.relaxation_parameter = 2 * self.relaxation_parameter
                    break
                x_vec.array_w[:] = x_copy.array_r[:]

                # Update solution
                self._update_solution(self._dx, x_vec)

                self._compute_coefficients(x_vec, self._coeffs)
                # Update internal variables prior to computing residual
                try:
                    self._post_solve(x_vec)
                except AttributeError:
                    pass

                # Compute residual (F)
                self._compute_residual(x_vec, self._b, self._coeffs)
                res_new = self._b.norm(PETSc.NormType.NORM_2)  # type: ignore
                # print(res_new)

            res_old = res_new
            if write_solution:
                if u is not None:
                    out.x.array[:] = x.x.array[:] + u.x.array[:]
                else:
                    out.x.array[:] = x.x.array[:]

                vtx.write(self.iteration)

            # Initialize initial residual
            if self.iteration == 1:
                self.initial_residual = self._dx.norm(PETSc.NormType.NORM_2)  # type: ignore

            # Test for convergence
            if self.convergence_criterion == ConvergenceCriterion.residual:
                self.residual, newton_converged = self._check_convergence(self._b)
            elif self.convergence_criterion == ConvergenceCriterion.incremental:
                # Subtract 1 to make sure initial residual is properly set
                if self.iteration == 1:
                    self.residual = 1
                    newton_converged = False
                else:
                    self.residual, newton_converged = self._check_convergence(self._dx)
            else:
                raise RuntimeError("Unknown convergence criterion")
        if write_solution:
            vtx.close()
        if newton_converged:
            if self.comm.rank == 0:
                print(
                    f"Newton solver finished in {self.iteration} iterations and ",
                    f" {self.krylov_iterations} linear solver iterations",
                    flush=True,
                )
        else:
            if self.error_on_nonconvergence:
                if self.iteration == self.max_it:
                    raise RuntimeError(
                        "Newton solver did not converge because maximum number ",
                        f"of iterations ({self.max_it}) was reached",
                    )
                else:
                    raise RuntimeError("Newton solver did not converge")
            else:
                print("Newton Solver did non converge", flush=True)
        t.stop()
        return self.iteration, newton_converged

    def __del__(self):
        pass
        # if self.krylov_solver is not None:
        #     self.krylov_solver.destroy()
