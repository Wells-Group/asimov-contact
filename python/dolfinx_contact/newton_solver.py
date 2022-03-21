# Copyright (C) 2022 Jørgen S. Dokken
#
# SPDX-License-Identifier:    MIT

from enum import Enum
from typing import Callable, Tuple, Union

import numpy
from dolfinx import fem
from mpi4py import MPI
from petsc4py import PETSc

__all__ = ["NewtonSolver", "ConvergenceCriterion"]


class ConvergenceCriterion(Enum):
    residual = 10
    incremental = 20


class NewtonSolver():
    __slots__ = ["max_it", "rtol", "atol", "report", "error_on_nonconvergence",
                 "convergence_criterion", "relaxation_parameter", "_compute_residual",
                 "_compute_jacobian", "_compute_preconditioner", "krylov_iterations",
                 "iteration", "residual", "initial_residual", "krylov_solver", "_dx", "comm",
                 "_A", "_b", "_P"]

    def __init__(self, comm: MPI.Comm, J: PETSc.Mat, b: PETSc.Vec):
        """
        Create a Newton solver

        Args:
            comm: The MPI communicator
            J: The matrix to assemble the Jacobian into
            b: The vector to assemble the residual into
        """

        self.max_it = 50
        self.rtol = 1e-9
        self.atol = 1e-10
        self.iteration = 0
        self.krylov_iterations = 0
        self.initial_residual = 0
        self.residual = 0
        self.comm = comm
        self.convergence_criterion = ConvergenceCriterion.residual
        self._A = J
        self._b = b
        self.krylov_solver = PETSc.KSP()
        self.krylov_solver.create(self.comm)
        self.krylov_solver.setOptionsPrefix("nls_solve_")

    def set_krylov_options(self, options: dict[str, str]):
        """
        Set options for Krylov solver
        """
        # Options that has to apply to all matrices, not just the solver matrix
        keys = ["matptap_via"]
        g_opts = {}
        opts = PETSc.Options()
        opts.prefixPush(self.krylov_solver.getOptionsPrefix())
        for k, v in options.items():
            if k in keys:
                g_opts[k] = v
            else:
                opts[k] = v
        opts.prefixPop()
        for k, v in g_opts.items():
            opts[k] = v
        self.krylov_solver.setFromOptions()
        self._A.setOptionsPrefix(self.krylov_solver.getOptionsPrefix())
        self._A.setFromOptions()
        self._b.setOptionsPrefix(self.krylov_solver.getOptionsPrefix())
        self._b.setFromOptions()

    def solve(self, u: Union[fem.Function, PETSc.Vec]):
        """
        Solve non-linear problem into function u.
        Returns the number of iterations and if the solver converged
        """
        try:
            n, converged = self._solve(u.vector)
            u.x.scatter_forward()
        except AttributeError:
            n, converged = self._solve(u)
            u.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        return n, converged

    @property
    def A(self) -> PETSc.Mat:
        """Get the Jacobian matrix"""
        return self._A

    @property
    def b(self) -> PETSc.Vec:
        """Get the residual vector"""
        return self._b

    def setJ(self, J: Callable[[PETSc.Vec, PETSc.Mat], None]):
        """
        Set the function for computing the Jacobian
        Args:
            J: Function to compute the Jacobian matrix.
        """
        self._compute_jacobian = J

    def setF(self, F: Callable[[PETSc.Vec, PETSc.Vec], None]):
        """
        Set the function for computing the residual
        Args:
            J: Function to compute the residual
        """
        self._compute_residual = F

    def setP(self, P: Callable[[PETSc.Vec, PETSc.Mat], None], Pmat: PETSc.Mat):
        """
        Set the function for computing the preconditioner matrix
        Args:
            P: Function to compute the preconditioner matrix b (x, P)
            Pmat: The matrix to assemble the preconditioner into
        """
        self._compute_preconditioner = P
        self._P = Pmat

    def _pre_computation(self, x: PETSc.Vec):
        x.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

    def _post_solve(self, x: PETSc.Vec):
        x.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

    def _check_convergence(self, r: PETSc.Vec):
        residual = r.norm(PETSc.NormType.NORM_2)
        try:
            relative_residual = residual / self.initial_residual
        except ZeroDivisionError:
            relative_residual = numpy.inf
        if self.comm.rank == 0:
            print(f"Newton Iteration {self.iteration}: r (abs) {residual} (atol={self.atol})",
                  flush=True, end=" ")
            print(f"r (rel) {relative_residual} (rtol={self.rtol})", flush=True)
        return residual, relative_residual < self.rtol or residual < self.atol

    def _update_solution(self, dx: PETSc.Vec, x: PETSc.Vec):
        """
        Compute x-= relaxation_patameter*dx
        """
        x.axpy(-self.relaxation_parameter, dx)

    def _solve(self, x: Union[PETSc.Vec, fem.Function]) -> Tuple[int, int]:
        try:
            x_vec = x.vector
        except AttributeError:
            x_vec = x

        # Reset iteration counts
        self.iteration = 0
        self.krylov_iterations = 0
        self.residual = -1

        try:
            self._pre_computation(x_vec)
        except AttributeError:
            raise RuntimeError("Pre-computation has not been set")

        try:
            self._compute_residual(x_vec, self._b)
        except AttributeError:
            raise RuntimeError("Function for computing residual vector has not been provided")

        newton_converged = False

        if self.convergence_criterion == ConvergenceCriterion.residual:
            self.residual, newton_converged = self._check_convergence(self.b)
        elif (self.convergence_criterion == ConvergenceCriterion.incremental):
            # We need to do at least one Newton step with the ||dx||-stopping criterion
            newton_converged = False
        else:
            raise ValueError("Unknown convergence criterion")

        try:
            self.krylov_solver.setOperators(self._A, self._P)
        except AttributeError:
            self.krylov_solver.setOperators(self._A, self._A)

        try:
            self._dx
        except AttributeError:
            self._dx = self._A.createVecRight()

        # Start iterations
        while not newton_converged and self.iteration < self.max_it:
            try:
                self._compute_jacobian(x_vec, self._A)
            except AttributeError:
                raise RuntimeError("Function for computing Jacobian has not been provided")

            try:
                self._compute_preconditioner(x_vec, self._P)
            except AttributeError:
                pass

            # Perform linear solve and update number of Krylov iterations
            self.krylov_solver.solve(self._b, self._dx)
            self.krylov_iterations += self.krylov_solver.getIterationNumber()

            # Update solution
            self._update_solution(self._dx, x_vec)

            # Increment iteration count
            self.iteration += 1

            # Update internal variables prior to computing residual
            try:
                self._post_solve(x_vec)
            except AttributeError:
                pass

            # Compute residual (F)
            self._compute_residual(x_vec, self._b)

            # Initialize initial residual
            if self.iteration == 1:
                self.initial_residual = self._dx.norm(PETSc.NormType.NORM_2)

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

        if newton_converged:
            if self.comm.rank == 0:
                print(f"Newton solver finished in {self.iteration} iterations and ",
                      f" {self.krylov_iterations} linear solver iterations", flush=True)
        else:
            if self.error_on_nonconvergence:
                if self.iteration == self.max_it:
                    raise RuntimeError("Newton solver did not converge because maximum number ",
                                       f"of iterations ({self.max_it}) was reached")
                else:
                    raise RuntimeError("Newton solver did not converge")
            else:
                print("Newton Solver did non converge", flush=True)
        return self.iteration, newton_converged