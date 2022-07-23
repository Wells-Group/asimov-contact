# Copyright (C) 2021 Sarah Roggendorf and Jørgen S. Dokken
#
# SPDX-License-Identifier:    MIT

"""User interface for contact"""

# flake8: noqa


from dolfinx_contact.cpp import (Kernel, QuadratureRule,
                                 compute_active_entities, pack_circumradius,
                                 update_geometry)

from .helpers import epsilon, lame_parameters, sigma_func, compare_matrices
from .newton_solver import ConvergenceCriterion, NewtonSolver

__all__ = ["NewtonSolver", "ConvergenceCriterion", "lame_parameters", "epsilon",
           "sigma_func", "Kernel", "pack_circumradius", "update_geometry",
           "QuadratureRule", "compute_active_entities", "compare_matrices"]
