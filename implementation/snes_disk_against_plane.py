import dolfinx
import dolfinx.fem
import dolfinx.io
import dolfinx.log
import dolfinx.mesh
import numpy as np
import ufl
from mpi4py import MPI
from petsc4py import PETSc
from IPython import embed

from helpers import NonlinearPDEProblem, NonlinearPDE_SNESProblem, lame_parameters, epsilon, sigma_func

def snes_solver(strain):

    # read in mesh
    with dolfinx.io.XDMFFile(MPI.COMM_WORLD, "disk.xdmf", "r") as xdmf:
        mesh = xdmf.read_mesh(name="Grid")

    #facet markers for different part of boundary
    def upper_part(x):
        return x[1] > 0.9

    def lower_part(x):
        return x[1] < 0.25

    tdim = mesh.topology.dim
    top_facets = dolfinx.mesh.locate_entities_boundary(mesh, tdim - 1, upper_part)
    bottom_facets = dolfinx.mesh.locate_entities_boundary(mesh, tdim - 1, lower_part)
    top_values = np.full(len(top_facets), 1, dtype=np.int32)
    bottom_values = np.full(len(bottom_facets), 2, dtype=np.int32)
    indices = np.concatenate([top_facets, bottom_facets])
    values = np.hstack([top_values, bottom_values])
    facet_marker = dolfinx.MeshTags(mesh, tdim - 1, indices, values)

    # write mesh and facet markers to xdmf
    with dolfinx.io.XDMFFile(MPI.COMM_WORLD, "results/mf.xdmf", "w") as xdmf:
        xdmf.write_mesh(mesh)
        xdmf.write_meshtags(facet_marker)

    # function space and problem parameters
    V = dolfinx.VectorFunctionSpace(mesh, ("CG", 1)) # function space 
    n = ufl.FacetNormal(mesh) # unit normal 
    E = 1e3   #young's modulus
    nu = 0.1  #poisson ratio
    h = ufl.Circumradius(mesh) #mesh size
    mu_func, lambda_func = lame_parameters(strain)
    mu = mu_func(E, nu)
    lmbda = lambda_func(E, nu)
    sigma = sigma_func(mu, lmbda)

    # Mimicking the plane y=-g
    g = 0.02

    # Functions for penalty term. Not used at the moment.
    # def gap(u): # Definition of gap function
    #     x = ufl.SpatialCoordinate(mesh)
    #     return x[1]+u[1]-g
    # def maculay(x): # Definition of Maculay bracket
    #     return (x+abs(x))/2
    

    # elasticity variational formulation no contact
    u = dolfinx.Function(V)
    v = ufl.TestFunction(V)
    dx = ufl.Measure("dx", domain=mesh)
    ds = ufl.Measure("ds", domain=mesh, subdomain_data=facet_marker)
    penalty = 0
    #F = ufl.inner(sigma(u), epsilon(v)) * dx - ufl.inner(dolfinx.Constant(mesh, (0, 0)), v) * dx
    # Stored strain energy density (linear elasticity model)
    psi = 1/2*ufl.inner(sigma(u), epsilon(u))
    Pi = psi*dx #+ 1/2*(penalty*E/h)*ufl.inner(maculay(-gap(u)),maculay(-gap(u)))*ds(1)

    # Compute first variation of Pi (directional derivative about u in the direction of v)
    F = ufl.derivative(Pi, u, v)

    # Dirichlet boundary conditions
    def _u_D(x):
        values = np.zeros((mesh.geometry.dim, x.shape[1]))
        values[0] = 0
        values[1] = -0.08
        return values
    u_D = dolfinx.Function(V)
    u_D.interpolate(_u_D)
    u_D.name = "u_D"
    u_D.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
    bc = dolfinx.DirichletBC(u_D, dolfinx.fem.locate_dofs_topological(V, tdim - 1, top_facets))
    bcs = [bc] 
    #create nonlinear problem
    problem = NonlinearPDE_SNESProblem (F, u, bc)  

    # Inequality constraints (contact constraints)
    # The displacement u must be such that the current configuration x+u
    # remains in the box [xmin = -inf,xmax = inf] x [ymin = -g,ymax = inf]
    # inf replaced by large number for implementation
    xmax = 1e7
    xmin = -1e7
    ymax = 1e7
    ymin = -g
    def _constraint_u(x):
        values = np.zeros((mesh.geometry.dim, x.shape[1]))
        values[0] = xmax - x[0]
        values[1] = ymax - x[1]
        return values

    def _constraint_l(x):
        values = np.zeros((mesh.geometry.dim, x.shape[1]))
        values[0] = xmin - x[0]
        values[1] = ymin - x[1]
        return values

    umax = dolfinx.Function(V)
    umax.interpolate(_constraint_u)
    umin = dolfinx.Function(V)
    umin.interpolate(_constraint_l)


    # Create semismooth Newton solver (SNES)
    b = dolfinx.cpp.la.create_vector(V.dofmap.index_map, V.dofmap.index_map_bs)
    J = dolfinx.cpp.fem.create_matrix(problem.a_comp._cpp_object)
    snes = PETSc.SNES().create()
    opts = PETSc.Options()
    opts["snes_monitor"] = None
    #opts["snes_view"] = None
    opts["snes_max_it"] =50
    opts["snes_no_convergence_test"] = False
    opts["snes_max_fail"]= 10
    opts["snes_type"] = "vinewtonrsls"
    opts["snes_rtol"] = 1e-9
    opts["snes_atol"] = 1e-9
    opts["snes_linear_solver"] = "lu"
    snes.setFromOptions()
    snes.setFunction(problem.F,b)
    snes.setJacobian(problem.J,J)
    snes.setVariableBounds(umin.vector, umax.vector)
    def _u_initial(x):
        values = np.zeros((mesh.geometry.dim, x.shape[1]))
        values[0] = 0
        values[1] = -0.01-g
        return values
    u.interpolate(_u_initial)
    dolfinx.log.set_log_level(dolfinx.log.LogLevel.INFO)
    snes.solve(None, u.vector)
    u.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

    assert(snes.getConvergedReason()>1)
    assert(snes.getConvergedReason()<4)
    with dolfinx.io.XDMFFile(MPI.COMM_WORLD, "results/u_snes.xdmf", "w") as xdmf:
        xdmf.write_mesh(mesh)
        xdmf.write_function(u)


if __name__ == "__main__":
    snes_solver(True)
    

