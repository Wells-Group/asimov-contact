from typing import Optional, Tuple
from dolfinx import common, cpp, fem
from dolfinx import mesh as _mesh
import numpy as np
import numpy.typing as npt
import ufl
from dolfinx.cpp.graph import AdjacencyList_int32

import dolfinx_contact
import dolfinx_contact.cpp

from .nitsche_unbiased import setup_newton_solver, get_problem_parameters
from dolfinx_contact.helpers import sigma_func


class ContactProblem:
    __slots__ = ["F", "J", "bcs", "u", "du", "contact", "markers", "entities",
                 "q_deg", "coeffs", "consts", "search_method", "newton_options",
                 "petsc_options", "coulomb", "normals"]

    def __init__(self, F: ufl.Form, J: ufl.Form,
                 bcs: list[fem.DirichletBC],
                 u: fem.Function, du: fem.Function, contact: dolfinx_contact.cpp.Contact,
                 markers: list[_mesh.MeshTags], entities: list[npt.NDArray[np.int32]],
                 quadrature_degree: int, const_coeffs: list[npt.NDArray[np.float64]],
                 consts: npt.NDArray[np.float64], search_method: list[dolfinx_contact.cpp.ContactMode],
                 petsc_options: Optional[dict] = None, newton_options: Optional[dict] = None, coulomb: bool = False,
                 normals: Optional[list[npt.NDArray[np.float64]]] = None):

        self.F = F
        self.J = J
        self.bcs = bcs
        self.u = u
        self.du = du
        self.contact = contact
        self.markers = markers
        self.entities = entities
        self.q_deg = quadrature_degree
        self.coeffs = const_coeffs
        self.consts = consts
        self.search_method = search_method
        self.newton_options = newton_options
        self.petsc_options = petsc_options
        self.coulomb = coulomb
        self.normals = normals

    def solve(self):
        newton_solver = setup_newton_solver(self.F, self.J, self.bcs, self.u, self.du, self.contact, self.markers,
                                            self.entities, self.q_deg, self.coeffs, self.consts,
                                            self.search_method, self.coulomb, self.normals)
        # Set Newton solver options
        newton_solver.set_newton_options(self.newton_options)
        print(self.newton_options)

        # Set Krylov solver options
        newton_solver.set_krylov_options(self.petsc_options)
        n, converged = newton_solver.solve(self.du)
        if not converged:
            print("Newton solver did not converge")
        return n

    def set_normals(self):
        normals = []
        for i in range(len(self.normals)):
            if self.search_method[i] == dolfinx_contact.cpp.ContactMode.Raytracing:
                normals.append(-self.contact.pack_nx(i))
            else:
                normals.append(self.contact.pack_ny(i))
        self.normals = normals


def create_contact_solver(ufl_form: ufl.Form, u: fem.Function,
                          mu: fem.Function, lmbda: fem.Function,
                          markers: list[_mesh.MeshTags],
                          contact_data: Tuple[AdjacencyList_int32, list[Tuple[int, int]]],
                          bcs: list[fem.DirichletBC],
                          problem_parameters: dict[str, np.float64],
                          search_method: list[dolfinx_contact.cpp.ContactMode],
                          quadrature_degree: int = 5,
                          form_compiler_options: Optional[dict] = None,
                          jit_options: Optional[dict] = None,
                          petsc_options: Optional[dict] = None,
                          newton_options: Optional[dict] = None,
                          search_radius: np.float64 = np.float64(-1.),
                          coulomb: bool = False, dt=1.0) -> ContactProblem:
    """
    Use custom kernel to compute the contact problem with two elastic bodies coming into contact.

    Parameters
    ==========
    ufl_form: The variational form without contact contribution
    u:        The function to be solved for. Also serves as initial value.
    markers
        A list of meshtags. The first element must mark all separate objects in order to create the correct nullspace.
        The second element must contain the mesh_tags for all puppet surfaces,
        Dirichlet-surfaces and Neumann-surfaces
        All further elements may contain candidate_surfaces
    contact_data = (surfaces, contact_pairs), where
        surfaces: Adjacency list. Links of i are meshtag values for contact
                  surfaces in ith mesh_tag in mesh_tags
        contact_pairs: list of pairs (i, j) marking the ith surface as a puppet
                  surface and the jth surface as the corresponding candidate
                  surface
    problem_parameters
        Dictionary with lame parameters and Nitsche parameters.
        Valid (key, value) tuples are: ('gamma': float), ('theta', float), ('mu', float),
        (lambda, float),
        where theta can be -1, 0 or 1 for skew-symmetric, penalty like or symmetric
        enforcement of Nitsche conditions
    quadrature_degree
        The quadrature degree to use for the custom contact kernels
    form_compiler_options
        Parameters used in FFCX compilation of this form. Run `ffcx --help` at
        the commandline to see all available options. Takes priority over all
        other parameter values, except for `scalar_type` which is determined by
        DOLFINX.
    jit_options
        Parameters used in CFFI JIT compilation of C code generated by FFCX.
        See https://github.com/FEniCS/dolfinx/blob/main/python/dolfinx/jit.py
        for all available parameters. Takes priority over all other parameter values.
    petsc_options
        Parameters that is passed to the linear algebra backend
        PETSc. For available choices for the 'petsc_options' kwarg,
        see the `PETSc-documentation
        <https://petsc4py.readthedocs.io/en/stable/manual/ksp/>`
    newton_options
        Dictionary with Newton-solver options. Valid (key, item) tuples are:
        ("atol", float), ("rtol", float), ("convergence_criterion", "str"),
        ("max_it", int), ("error_on_nonconvergence", bool), ("relaxation_parameter", float)

    """
    form_compiler_options = {} if form_compiler_options is None else form_compiler_options
    jit_options = {} if jit_options is None else jit_options
    petsc_options = {} if petsc_options is None else petsc_options
    newton_options = {} if newton_options is None else newton_options
    theta, gamma, s = get_problem_parameters(problem_parameters)
    sigma = sigma_func(mu, lmbda)

    # Contact data
    contact_pairs = contact_data[1]
    contact_surfaces = contact_data[0]

    # Mesh, function space and FEM functions
    V = u.function_space
    mesh = V.mesh
    V2 = fem.FunctionSpace(mesh, ("DG", 0))
    v = ufl_form.arguments()[0]  # Test function
    w = ufl.TrialFunction(V)     # Trial function
    du = fem.Function(V)
    du.x.array[:] = u.x.array[:]
    u.x.array[:].fill(0)
    h = fem.Function(V2)
    tdim = mesh.topology.dim
    ncells = mesh.topology.index_map(tdim).size_local
    h_vals = cpp.mesh.h(mesh._cpp_object, mesh.topology.dim, np.arange(0, ncells, dtype=np.int32))
    h.x.array[:ncells] = h_vals[:]
    n = ufl.FacetNormal(mesh)

    # Integration measure and ufl part of linear/bilinear form
    ds = ufl.Measure("ds", domain=mesh, subdomain_data=markers[1])

    # ufl part of contact
    for contact_pair in contact_pairs:
        surface_value = int(contact_surfaces.links(0)[contact_pair[0]])
        ufl_form += - 0.5 * theta * h / gamma * ufl.inner(sigma(u) * n, sigma(v) * n) * \
            ds(surface_value)
    F = ufl.replace(ufl_form, {u: u + du})
    J = ufl.derivative(F, du, w)

    # compiled forms for rhs and tangen system
    F_custom = fem.form(F, form_compiler_options=form_compiler_options, jit_options=jit_options)
    J_custom = fem.form(J, form_compiler_options=form_compiler_options, jit_options=jit_options)

    # create contact class
    markers_cpp = [marker._cpp_object for marker in markers[1:]]
    with common.Timer("~Contact: Init"):
        contact = dolfinx_contact.cpp.Contact(markers_cpp, contact_surfaces, contact_pairs,
                                              V._cpp_object, quadrature_degree=quadrature_degree,
                                              search_method=search_method)

    contact.set_search_radius(search_radius)

    # pack constants
    consts = np.array([gamma, theta, dt], dtype=np.float64)

    # Retrieve active entities
    entities = []
    with common.Timer("~Contact: Compute active entities"):
        for pair in contact_pairs:
            entities.append(contact.active_entities(pair[0]))

    # interpolate friction coefficient
    fric_coeff = fem.Function(V2)
    fric_coeff.interpolate(lambda x: np.full((1, x.shape[1]), s))

    # Pack material parameters
    material = []
    with common.Timer("~Contact: Pack coeffs (mu, lmbda"):
        for i in range(len(contact_pairs)):
            material.append(np.hstack([dolfinx_contact.cpp.pack_coefficient_quadrature(
                mu._cpp_object, 0, entities[i]),
                dolfinx_contact.cpp.pack_coefficient_quadrature(
                lmbda._cpp_object, 0, entities[i]),
                dolfinx_contact.cpp.pack_coefficient_quadrature(
                fric_coeff._cpp_object, 0, entities[i])]))

    # Pack celldiameter on each surface
    h_packed = []
    with common.Timer("~Contact: Compute and pack celldiameter"):
        surface_cells = np.unique(np.hstack([entities[i][:, 0] for i in range(len(contact_pairs))]))
        h_int = fem.Function(V2)
        expr = fem.Expression(h, V2.element.interpolation_points())
        h_int.interpolate(expr, surface_cells)
        for i in range(len(contact_pairs)):
            h_packed.append(dolfinx_contact.cpp.pack_coefficient_quadrature(
                h_int._cpp_object, 0, entities[i]))

    for j in range(len(contact_pairs)):
        contact.create_distance_map(j)
    normals = []
    for i in range(len(contact_pairs)):
        if search_method[i] == dolfinx_contact.cpp.ContactMode.Raytracing:
            normals.append(-contact.pack_nx(i))
        else:
            normals.append(contact.pack_ny(i))

    # Concatenate material parameters, he4
    const_coeffs = []
    for i in range(len(contact_pairs)):
        const_coeffs.append(np.hstack([material[i], h_packed[i]]))

    problem = ContactProblem(F_custom, J_custom, bcs, u, du, contact, markers, entities,
                             quadrature_degree, const_coeffs, consts, search_method,
                             petsc_options, newton_options, coulomb, normals)

    return problem
