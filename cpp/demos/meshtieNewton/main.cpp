// Copyright (C) 2023 Sarah Roggendorf
//
// This file is part of DOLFINx_Contact
//
// SPDX-License-Identifier:    MIT

// meshtie demo written as a nonlinear problem using
// the NewtonSolver provided by dolfinx
// ====================================================

#include "linear_elasticity.h"
#include <dolfinx.h>
#include <dolfinx/common/log.h>
#include <dolfinx/fem/CoordinateElement.h>
#include <dolfinx/io/XDMFFile.h>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/mesh/Topology.h>
#include <dolfinx/mesh/utils.h>
#include <dolfinx_contact/Contact.h>
#include <dolfinx_contact/MeshTie.h>
#include <dolfinx_contact/parallel_mesh_ghosting.h>
#include <dolfinx_contact/utils.h>

using T = PetscScalar;
using U = typename dolfinx::scalar_value_type_t<T>;

//------------------------------------------------------------------------------
/// Problem class to define a mesh tying problem as a non-linear problem
//------------------------------------------------------------------------------
class MeshTieProblem
{
public:
  /// Constructor
  /// @param[in] L The form describing the residual
  /// @param[in] J The bilinear form describint the tangent system
  /// @param[in] bcs The boundary conditions
  /// @param[in] meshties The MeshTie class describing the tied surfaces
  /// @param[in] subdomains The domain marker labelling the individual
  /// components
  /// @param[in] u The displacement function to be solved for
  /// @param[in] lmbda The lame parameter lambda
  /// @param[in] mu The lame parameter mu
  /// @param[in] gamma Nitsche parameter
  /// @param[in] theta parameter selecting version of Nitsche (1 - symmetric, -1
  /// anti-symmetric, 0 - penalty-like)
  MeshTieProblem(
      std::shared_ptr<dolfinx::fem::Form<T>> L,
      std::shared_ptr<dolfinx::fem::Form<T>> J,
      std::vector<std::shared_ptr<const dolfinx::fem::DirichletBC<T>>> bcs,
      std::shared_ptr<dolfinx_contact::MeshTie> meshties,
      dolfinx::mesh::MeshTags<std::int32_t> subdomains,
      std::shared_ptr<dolfinx::fem::Function<T>> u,
      std::shared_ptr<dolfinx::fem::Function<T>> lmbda,
      std::shared_ptr<dolfinx::fem::Function<T>> mu, double gamma, double theta)
      : _l(L), _j(J), _bcs(bcs), _meshties(meshties),
        _b(L->function_spaces()[0]->dofmap()->index_map,
           L->function_spaces()[0]->dofmap()->index_map_bs()),
        _matA(dolfinx::la::petsc::Matrix(
            meshties->create_petsc_matrix(*J, std::string()), false)),
        _u(u)
  {
    // create PETSc rhs vector
    auto map = L->function_spaces()[0]->dofmap()->index_map;
    const int bs = L->function_spaces()[0]->dofmap()->index_map_bs();
    std::int32_t size_local = bs * map->size_local();
    std::vector<PetscInt> ghosts(map->ghosts().begin(), map->ghosts().end());
    std::int64_t size_global = bs * map->size_global();
    VecCreateGhostBlockWithArray(map->comm(), bs, size_local, size_global,
                                 ghosts.size(), ghosts.data(),
                                 _b.array().data(), &_b_petsc);

    // initialise the input data for integration kernels
    _meshties->generate_kernel_data(
        dolfinx_contact::Problem::Elasticity, L->function_spaces()[0],
        {{"u", u}, {"mu", mu}, {"lambda", lmbda}}, gamma, theta);

    // build near null space preventing rigid body motion of individual
    // components)
    std::vector<std::int32_t> tags = {1, 2};
    MatNullSpace ns = dolfinx_contact::build_nullspace_multibody(
        *L->function_spaces()[0], subdomains, tags);
    MatSetNearNullSpace(_matA.mat(), ns);
    MatNullSpaceDestroy(&ns);
  }

  /// Destructor
  virtual ~MeshTieProblem()
  {
    if (_b_petsc)
      VecDestroy(&_b_petsc);
  }

  /// Define function that is called by the NewtonSolver before the residual or
  /// Jacobian are computed.
  //// Used to update ghost values.
  auto form()
  {
    return [](Vec x)
    {
      VecGhostUpdateBegin(x, INSERT_VALUES, SCATTER_FORWARD);
      VecGhostUpdateEnd(x, INSERT_VALUES, SCATTER_FORWARD);
    };
  }

  /// Compute residual F at current point x
  auto F()
  {
    return [&](const Vec x, Vec bin)
    {
      // Avoid long log output
      // loguru::g_stderr_verbosity = loguru::Verbosity_OFF;

      // Generate input data for custom kernel
      _meshties->update_kernel_data({{"u", _u}},
                                    dolfinx_contact::Problem::Elasticity);

      // Assemble b
      std::span<T> b(_b.mutable_array());
      std::fill(b.begin(), b.end(), 0.0);
      _meshties->assemble_vector(
          b, _l->function_spaces()[0],
          dolfinx_contact::Problem::Elasticity); // custom kernel for mesh tying
      fem::assemble_vector<T>(b, *_l);           // standard assembly

      // Apply lifting
      Vec x_local;
      VecGhostGetLocalForm(x, &x_local);
      PetscInt n = 0;
      VecGetSize(x_local, &n);
      const T* array = nullptr;
      VecGetArrayRead(x_local, &array);
      dolfinx::fem::apply_lifting<T, U>(
          b, {_j}, {_bcs}, {std::span<const T>(array, n)}, T(-1.0));

      // scatter reverse
      VecGhostUpdateBegin(_b_petsc, ADD_VALUES, SCATTER_REVERSE);
      VecGhostUpdateEnd(_b_petsc, ADD_VALUES, SCATTER_REVERSE);

      // Set bc
      fem::set_bc<T, U>(b, _bcs, std::span<const T>(array, n), -1.0);
      VecRestoreArrayRead(x, &array);

      // log level INFO ensures Newton steps are logged
      // loguru::g_stderr_verbosity = loguru::Verbosity_INFO;
    };
  }

  /// Compute tangent system matrix A
  auto J()
  {
    return [&](const Vec, Mat A)
    {
      // Avoid long log output
      // loguru::g_stderr_verbosity = loguru::Verbosity_OFF;

      // Set matrix to 0
      MatZeroEntries(A);

      // custom assembly for mesh tying
      _meshties->assemble_matrix(la::petsc::Matrix::set_block_fn(A, ADD_VALUES),
                                 _j->function_spaces()[0],
                                 dolfinx_contact::Problem::Elasticity);
      MatAssemblyBegin(A, MAT_FLUSH_ASSEMBLY);
      MatAssemblyEnd(A, MAT_FLUSH_ASSEMBLY);

      // standard assembly
      dolfinx::fem::assemble_matrix(
          dolfinx::la::petsc::Matrix::set_block_fn(A, ADD_VALUES), *_j, _bcs);
      MatAssemblyBegin(A, MAT_FLUSH_ASSEMBLY);
      MatAssemblyEnd(A, MAT_FLUSH_ASSEMBLY);

      // set diagonal for bcs
      dolfinx::fem::set_diagonal<T>(
          dolfinx::la::petsc::Matrix::set_fn(A, INSERT_VALUES),
          *_u->function_space(), _bcs);
      MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
      MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);

      // log level INFO ensures Newton steps are logged
      // loguru::g_stderr_verbosity = loguru::Verbosity_INFO;
    };
  }

  /// return the rhs PETSc vector
  Vec vector() { return _b_petsc; }

  /// return the system matrix
  Mat matrix() { return _matA.mat(); }

private:
  // forms describing residual and sytem matrix
  std::shared_ptr<dolfinx::fem::Form<T>> _l, _j;
  // boundary conditions
  std::vector<std::shared_ptr<const dolfinx::fem::DirichletBC<T>>> _bcs;
  // MeshTie class describing tied surfaces
  std::shared_ptr<dolfinx_contact::MeshTie> _meshties;
  // rhs vector
  dolfinx::la::Vector<T> _b;
  Vec _b_petsc = nullptr;
  // system matrix
  la::petsc::Matrix _matA;
  // displacement function
  std::shared_ptr<dolfinx::fem::Function<T>> _u;
};

int main(int argc, char* argv[])
{

  init_logging(argc, argv);

  PetscInitialize(&argc, &argv, nullptr, nullptr);
  // Set the logging thread name to show the process rank
  int mpi_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  std::string thread_name = "RANK " + std::to_string(mpi_rank);
  loguru::set_thread_name(thread_name.c_str());
  {
    auto [mesh_init, domain1_init, facet1_init]
        = dolfinx_contact::read_mesh("../meshes/cont-blocks_sk24_fnx.xdmf");

    const std::int32_t contact_bdry_1 = 12; // top contact interface
    const std::int32_t contact_bdry_2 = 6;  // bottom contact interface
    // loguru::g_stderr_verbosity = loguru::Verbosity_OFF;
    auto [mesh_new, facet1, domain1] = dolfinx_contact::create_contact_mesh(
        *mesh_init, facet1_init, domain1_init, {contact_bdry_1, contact_bdry_2},
        10.0);

    auto mesh = std::make_shared<dolfinx::mesh::Mesh<U>>(mesh_new);
    // Create function spaces
    auto ct = mesh->topology()->cell_type();
    auto element_mu = basix::create_element<double>(
        basix::element::family::P, dolfinx::mesh::cell_type_to_basix_type(ct),
        0, basix::element::lagrange_variant::unset,
        basix::element::dpc_variant::unset, true);
    auto element = basix::create_element<double>(
        basix::element::family::P, dolfinx::mesh::cell_type_to_basix_type(ct),
        1, basix::element::lagrange_variant::unset,
        basix::element::dpc_variant::unset, false);

    auto V = std::make_shared<fem::FunctionSpace<double>>(
        fem::create_functionspace(mesh, element,
                                  {(std::size_t)mesh->geometry().dim()}));
    auto V0 = std::make_shared<fem::FunctionSpace<double>>(
        fem::create_functionspace(mesh, element_mu));

    // Problem parameters (material & Nitsche)
    double E = 1E4;
    double nu = 0.2;
    double gamma = E * 10;
    double theta = 1;

    double lmbda_val = E * nu / ((1 + nu) * (1 - 2 * nu));
    double mu_val = E / (2 * (1 + nu));

    // Create DG0 function for lame parameter lambda
    auto lmbda = std::make_shared<fem::Function<T>>(V0);
    lmbda->interpolate(
        [lmbda_val](
            auto x) -> std::pair<std::vector<T>, std::vector<std::size_t>>
        {
          std::vector<T> _f;
          for (std::size_t p = 0; p < x.extent(1); ++p)
          {
            _f.push_back(lmbda_val);
          }
          return {_f, {_f.size()}};
        });

    // create DG0 function for lame parameter mu
    auto mu = std::make_shared<fem::Function<T>>(V0);
    mu->interpolate(
        [mu_val](auto x) -> std::pair<std::vector<T>, std::vector<std::size_t>>
        {
          std::vector<T> _f;
          for (std::size_t p = 0; p < x.extent(1); ++p)
          {
            _f.push_back(mu_val);
          }
          return {_f, {_f.size()}};
        });

    // create integration domains for integrating over specific surfaces
    auto facet_domains = fem::compute_integration_domains(
        fem::IntegralType::exterior_facet, *facet1.topology(), facet1.indices(),
        facet1.dim(), facet1.values());

    // diplacement function
    auto u = std::make_shared<fem::Function<T>>(V);

    // Define variational forms
    auto J = std::make_shared<fem::Form<T>>(
        fem::create_form<T>(*form_linear_elasticity_J, {V, V},
                            {{"mu", mu}, {"lmbda", lmbda}}, {}, {}));
    std::vector<std::pair<std::int32_t, std::span<const std::int32_t>>>
        integration_domain;
    std::transform(
        facet_domains.begin(), facet_domains.end(),
        std::back_inserter(integration_domain),
        [](auto& domain)
            -> std::pair<std::int32_t, std::span<const std::int32_t>> {
          return {domain.first, std::span<const std::int32_t>(domain.second)};
        });
    auto F = std::make_shared<fem::Form<T>>(fem::create_form<T>(
        *form_linear_elasticity_F, {V},
        {{"u", u}, {"mu", mu}, {"lmbda", lmbda}}, {},
        {{dolfinx::fem::IntegralType::exterior_facet, integration_domain}}));

    // Define boundary conditions
    // bottom fixed, top displaced in y-diretion by -0.2
    const std::int32_t dirichlet_bdy_1 = 8; // top face
    const std::int32_t dirichlet_bdy_2 = 2; // bottom face
    auto facets_1 = facet1.find(dirichlet_bdy_1);
    auto facets_2 = facet1.find(dirichlet_bdy_2);
    auto bdofs_1 = dolfinx::fem::locate_dofs_topological(
        *V->mesh()->topology_mutable(), *V->dofmap(), 2, facets_1);
    auto bdofs_2 = dolfinx::fem::locate_dofs_topological(
        *V->mesh()->topology_mutable(), *V->dofmap(), 2, facets_2);
    auto bcs = {std::make_shared<const dolfinx::fem::DirichletBC<T>>(
                    std::vector<T>({0.0, -0.2, 0.0}), bdofs_1, V),
                std::make_shared<const dolfinx::fem::DirichletBC<T>>(
                    std::vector<T>({0.0, 0.0, 0.0}), bdofs_2, V)};

    // Define meshtie data
    // point to correct surface markers
    std::vector<std::int32_t> data = {contact_bdry_1, contact_bdry_2};
    std::vector<std::int32_t> offsets = {0, 2};
    auto contact_markers
        = std::make_shared<dolfinx::graph::AdjacencyList<std::int32_t>>(
            std::move(data), std::move(offsets));
    // wrap facet markers
    std::vector<std::shared_ptr<dolfinx::mesh::MeshTags<std::int32_t>>> markers
        = {std::make_shared<dolfinx::mesh::MeshTags<std::int32_t>>(facet1)};
    // define pairs (slave, master)
    std::vector<std::array<int, 2>> pairs = {{0, 1}, {1, 0}};

    // create meshties
    auto meshties = std::make_shared<dolfinx_contact::MeshTie>(
        dolfinx_contact::MeshTie(markers, contact_markers, pairs, mesh, 5));

    // create "non-linear" meshtie problem (linear problem written as non-linear
    // problem)
    auto problem = MeshTieProblem(F, J, bcs, meshties, domain1, u, lmbda, mu,
                                  gamma, theta);

    // loglevel INFO shows NewtonSolver output
    // loguru::g_stderr_verbosity = loguru::Verbosity_INFO;

    // create Newton Solver
    dolfinx::nls::petsc::NewtonSolver newton_solver(mesh->comm());
    newton_solver.setF(problem.F(), problem.vector());
    newton_solver.setJ(problem.J(), problem.matrix());
    newton_solver.set_form(problem.form());
    newton_solver.rtol = 1E-7;
    newton_solver.atol = 1E-7;
    newton_solver.max_it = 50;

    // Set up linear solver with parameters
    dolfinx::la::petsc::KrylovSolver& ksp = newton_solver.get_krylov_solver();
    const std::string options_prefix = "nls_solve_";
    dolfinx::la::petsc::options::set(options_prefix + "ksp_type", "cg");
    dolfinx::la::petsc::options::set(options_prefix + "pc_type", "gamg");
    dolfinx::la::petsc::options::set(options_prefix + "mg_levels_ksp_type",
                                     "chebyshev");
    dolfinx::la::petsc::options::set(options_prefix + "mg_levels_pc_type",
                                     "jacobi");
    dolfinx::la::petsc::options::set(options_prefix + "pc_gamg_type", "agg");
    dolfinx::la::petsc::options::set(options_prefix + "pc_gamg_coarse_eq_limit",
                                     100);
    dolfinx::la::petsc::options::set(options_prefix + "pc_gamg_agg_nsmooths",
                                     1);
    dolfinx::la::petsc::options::set(options_prefix + "pc_gamg_threshold",
                                     1E-3);
    dolfinx::la::petsc::options::set(options_prefix + "pc_gamg_square_graph",
                                     2);
    dolfinx::la::petsc::options::set(options_prefix + "ksp_rtol", 1E-10);
    dolfinx::la::petsc::options::set(options_prefix + "ksp_atol", 1E-10);
    dolfinx::la::petsc::options::set(options_prefix + "ksp_norm_type",
                                     "unpreconditioned");
    dolfinx::la::petsc::options::set(options_prefix + "ksp_monitor");
    ksp.set_from_options();

    // petsc vector corresponding to displacement function
    dolfinx::la::petsc::Vector _u(
        dolfinx::la::petsc::create_vector_wrap(*u->x()), false);

    // solve non-linear problem
    newton_solver.solve(_u.vec());

    // loguru::g_stderr_verbosity = loguru::Verbosity_OFF;
    // write solution to file
    dolfinx::io::XDMFFile file(mesh->comm(), "result.xdmf", "w");
    file.write_mesh(*mesh);
    file.write_function(*u, 0.0);
    file.close();
  }
  PetscFinalize();
  return 0;
}