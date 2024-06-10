// Copyright (C) 2023 Sarah Roggendorf
//
// This file is part of DOLFINx_Contact
//
// SPDX-License-Identifier:    MIT

// demo showing the meshtie capability for
// thermo-elasticity
// ====================================================

#include "thermo_elasticity.h"
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
/// Problem class to define the elastic part of the system as a non-linear
/// problem
//------------------------------------------------------------------------------
class ThermoElasticProblem
{
public:
  /// Constructor
  /// @param[in] L The form describing the residual
  /// @param[in] J The bilinear form describint the tangent system
  /// @param[in] bcs The boundary conditions
  /// @param[in] meshties The MeshTie class describing the tied surfaces
  /// @param[in] subdomains The domain marker labelling the individual
  /// components
  /// @param[in] subdomain_tags The tags of the individual components
  /// @param[in] u The displacement function to be solved for
  /// @param[in] T The temperature
  /// @param[in] lmbda The lame parameter lambda
  /// @param[in] mu The lame parameter mu
  /// @param[in] gamma Nitsche parameter
  /// @param[in] theta parameter selecting version of Nitsche (1 - symmetric, -1
  /// anti-symmetric, 0 - penalty-like)
  /// @param[in] alpha thermo-elastic coefficient
  ThermoElasticProblem(
      std::shared_ptr<dolfinx::fem::Form<T>> L,
      std::shared_ptr<dolfinx::fem::Form<T>> J,
      std::vector<std::shared_ptr<const dolfinx::fem::DirichletBC<T>>> bcs,
      std::shared_ptr<dolfinx_contact::MeshTie> meshties,
      dolfinx::mesh::MeshTags<std::int32_t> subdomains,
      std::vector<std::int32_t> subdomain_tags,
      std::shared_ptr<dolfinx::fem::Function<T>> u,
      std::shared_ptr<dolfinx::fem::Function<T>> T0,
      std::shared_ptr<dolfinx::fem::Function<T>> lmbda,
      std::shared_ptr<dolfinx::fem::Function<T>> mu,
      std::shared_ptr<dolfinx::fem::Function<T>> alpha, double gamma,
      double theta)
      : _l(L), _j(J), _bcs(bcs), _meshties(meshties),
        _b(L->function_spaces()[0]->dofmap()->index_map,
           L->function_spaces()[0]->dofmap()->index_map_bs()),
        _matA(dolfinx::la::petsc::Matrix(
            meshties->create_petsc_matrix(*J, std::string()), false)),
        _u(u), _T(T0)
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
        dolfinx_contact::Problem::ThermoElasticity, *L->function_spaces()[0],
        {{"u", u}, {"T", T0}, {"mu", mu}, {"lambda", lmbda}, {"alpha", alpha}},
        gamma, theta);

    // build near null space preventing rigid body motion of individual
    // components)
    MatNullSpace ns = dolfinx_contact::build_nullspace_multibody(
        *L->function_spaces()[0], subdomains, subdomain_tags);
    MatSetNearNullSpace(_matA.mat(), ns);
    MatNullSpaceDestroy(&ns);
  }

  /// Destructor
  virtual ~ThermoElasticProblem()
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
    return [&](const Vec x, Vec /*bin*/)
    {
      // Avoid long log output
      // loguru::g_stderr_verbosity = loguru::Verbosity_OFF;

      // Generate input data for custom kernel
      _meshties->update_kernel_data({{"u", _u}, {"T", _T}},
                                    dolfinx_contact::Problem::ThermoElasticity);

      // Assemble b
      std::span<T> b(_b.mutable_array());
      std::fill(b.begin(), b.end(), 0.0);
      _meshties->assemble_vector(
          b, *_l->function_spaces()[0],
          dolfinx_contact::Problem::ThermoElasticity); // custom kernel for mesh
                                                       // tying
      fem::assemble_vector<T>(b, *_l);                 // standard assembly

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
                                 *_j->function_spaces()[0],
                                 dolfinx_contact::Problem::ThermoElasticity);
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
  // temperature function
  std::shared_ptr<dolfinx::fem::Function<T>> _T;
};

int main(int argc, char* argv[])
{

  // const std::size_t time_steps = 40;
  init_logging(argc, argv);
  PetscInitialize(&argc, &argv, nullptr, nullptr);

  // Set the logging thread name to show the process rank
  int mpi_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  std::string fmt = "[%Y-%m-%d %H:%M:%S.%e] [RANK " + std::to_string(mpi_rank)
                    + "] [%l] %v";
  spdlog::set_pattern(fmt);
  {

    // Read in mesh
    auto [mesh_init, domain1_init, facet1_init]
        = dolfinx_contact::read_mesh("cont-blocks_sk24_fnx.xdmf");

    // Add necessary ghosts
    const std::int32_t contact_bdry_1 = 12; // top contact interface
    const std::int32_t contact_bdry_2 = 6;  // bottom contact interface
    // loguru::g_stderr_verbosity = loguru::Verbosity_OFF;
    auto [mesh_new, facet1, domain1] = dolfinx_contact::create_contact_mesh(
        *mesh_init, facet1_init, domain1_init, {contact_bdry_1, contact_bdry_2},
        10.0);
    auto mesh = std::make_shared<dolfinx::mesh::Mesh<U>>(mesh_new);

    // Find facets for boundary conditions (used for both the thermal and
    // elastic part)
    const std::int32_t dirichlet_bdy_1 = 8; // top face
    const std::int32_t dirichlet_bdy_2 = 2; // bottom face
    auto facets_1 = facet1.find(dirichlet_bdy_1);
    auto facets_2 = facet1.find(dirichlet_bdy_2);

    // Define meshtie data structures
    // point to correct surface markers
    std::vector<std::int32_t> data = {contact_bdry_1, contact_bdry_2};
    std::vector<std::int32_t> offsets = {0, 2};
    auto contact_markers
        = std::make_shared<dolfinx::graph::AdjacencyList<std::int32_t>>(
            std::move(data), std::move(offsets));
    // wrap facet markers
    std::vector<std::shared_ptr<const dolfinx::mesh::MeshTags<std::int32_t>>>
        markers
        = {std::make_shared<dolfinx::mesh::MeshTags<std::int32_t>>(facet1)};
    // define pairs (slave, master)
    std::vector<std::array<int, 2>> pairs = {{0, 1}, {1, 0}};

    // create meshties
    auto meshties = std::make_shared<dolfinx_contact::MeshTie>(
        dolfinx_contact::MeshTie(markers, *contact_markers, pairs, mesh, 5));

    // Nitsche parameters
    double gamma = 10;
    double theta = 1;

    // DG function space used for parameters
    auto ct = mesh->topology()->cell_type();
    auto element_mu = basix::create_element<double>(
        basix::element::family::P, dolfinx::mesh::cell_type_to_basix_type(ct),
        0, basix::element::lagrange_variant::unset,
        basix::element::dpc_variant::unset, true);
    auto element = basix::create_element<double>(
        basix::element::family::P, dolfinx::mesh::cell_type_to_basix_type(ct),
        1, basix::element::lagrange_variant::unset,
        basix::element::dpc_variant::unset, false);

    auto V0 = std::make_shared<fem::FunctionSpace<U>>(
        fem::create_functionspace(mesh, element_mu));

    // Thermal Problem
    auto Q = std::make_shared<fem::FunctionSpace<U>>(
        fem::create_functionspace(mesh, element));

    double kdt_val = 0.1;
    auto kdt = std::make_shared<fem::Function<T>>(V0);
    kdt->interpolate(
        [kdt_val](auto x) -> std::pair<std::vector<T>, std::vector<std::size_t>>
        {
          std::vector<T> _f;
          for (std::size_t p = 0; p < x.extent(1); ++p)
          {
            _f.push_back(kdt_val);
          }
          return {_f, {_f.size()}};
        });

    // temperature function
    auto T0 = std::make_shared<fem::Function<T>>(Q);

    // Define variational forms
    auto a_therm = std::make_shared<fem::Form<T>>(
        fem::create_form<T>(*form_thermo_elasticity_a_therm, {Q, Q},
                            {{"kdt", kdt}, {"T0", T0}}, {}, {}));
    auto L_therm = std::make_shared<fem::Form<T>>(
        fem::create_form<T>(*form_thermo_elasticity_L_therm, {Q},
                            {{"kdt", kdt}, {"T0", T0}}, {}, {}));

    // Define boundary conditions
    // temperature at bottom
    auto bdofs_therm = dolfinx::fem::locate_dofs_topological(
        *Q->mesh()->topology_mutable(), *Q->dofmap(), 2, facets_2);

    auto bcs_therm = std::make_shared<const dolfinx::fem::DirichletBC<T>>(
        1.0, bdofs_therm, Q);

    // Data for meshtie integration kernels
    meshties->generate_kernel_data(dolfinx_contact::Problem::Poisson, *Q,
                                   {{"kdt", kdt}, {"T", T0}}, gamma, theta);

    // Create matrix and vector
    auto A_therm = dolfinx::la::petsc::Matrix(
        meshties->create_petsc_matrix(*a_therm, std::string()), false);
    dolfinx::la::Vector<T> b_therm(
        L_therm->function_spaces()[0]->dofmap()->index_map,
        L_therm->function_spaces()[0]->dofmap()->index_map_bs());

    // Create linear solver for thermal problem
    dolfinx::la::petsc::KrylovSolver ksp_therm(MPI_COMM_WORLD);
    dolfinx::la::petsc::options::set("ksp_type", "cg");
    dolfinx::la::petsc::options::set("pc_type", "gamg");
    dolfinx::la::petsc::options::set("pc_mg_levels", 2);
    dolfinx::la::petsc::options::set("mg_levels_ksp_type", "chebyshev");
    dolfinx::la::petsc::options::set("mg_levels_pc_type", "jacobi");
    dolfinx::la::petsc::options::set("pc_gamg_type", "agg");
    dolfinx::la::petsc::options::set("pc_gamg_coarse_eq_limit", 100);
    dolfinx::la::petsc::options::set("pc_gamg_agg_nsmooths", 1);
    dolfinx::la::petsc::options::set("pc_gamg_threshold", 1E-3);
    dolfinx::la::petsc::options::set("pc_gamg_square_graph", 2);
    dolfinx::la::petsc::options::set("ksp_rtol", 1E-10);
    dolfinx::la::petsc::options::set("ksp_atol", 1E-10);
    dolfinx::la::petsc::options::set("ksp_norm_type", "unpreconditioned");
    dolfinx::la::petsc::options::set("ksp_monitor");

    ksp_therm.set_from_options();
    ksp_therm.set_operator(A_therm.mat());
    // petsc wrap for temperature and rhs
    dolfinx::la::petsc::Vector _T(
        dolfinx::la::petsc::create_vector_wrap(*T0->x()), false);
    dolfinx::la::petsc::Vector _b_therm(
        dolfinx::la::petsc::create_vector_wrap(b_therm), false);

    // Thermo-elastic problem
    // Create function space
    auto V = std::make_shared<fem::FunctionSpace<U>>(fem::create_functionspace(
        mesh, element, {(std::size_t)mesh->geometry().dim()}));

    // Problem parameters
    double E = 1E4;
    double nu = 0.2;
    double lmbda_val = E * nu / ((1 + nu) * (1 - 2 * nu));
    double mu_val = E / (2 * (1 + nu));
    double alpha = 0.3;
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

    auto alpha_c = std::make_shared<fem::Function<T>>(V0);
    alpha_c->interpolate(
        [alpha](auto x) -> std::pair<std::vector<T>, std::vector<std::size_t>>
        {
          std::vector<T> _f;
          for (std::size_t p = 0; p < x.extent(1); ++p)
          {
            _f.push_back(alpha);
          }
          return {_f, {_f.size()}};
        });

    // diplacement function
    auto u = std::make_shared<fem::Function<T>>(V);
    // Define variational forms
    auto J = std::make_shared<fem::Form<T>>(fem::create_form<T>(
        *form_thermo_elasticity_J, {V, V},
        {{"mu", mu}, {"lmbda", lmbda}, {"alpha", alpha_c}}, {}, {}));
    auto F = std::make_shared<fem::Form<T>>(
        fem::create_form<T>(*form_thermo_elasticity_F, {V},
                            {{"u", u},
                             {"T0", T0},
                             {"mu", mu},
                             {"lmbda", lmbda},
                             {"alpha", alpha_c}},
                            {}, {}));

    // Define boundary conditions
    // bottom fixed, top displaced in y-diretion by -0.2
    auto bdofs_1 = dolfinx::fem::locate_dofs_topological(
        *V->mesh()->topology_mutable(), *V->dofmap(), 2, facets_1);
    auto bdofs_2 = dolfinx::fem::locate_dofs_topological(
        *V->mesh()->topology_mutable(), *V->dofmap(), 2, facets_2);
    // bc_fun will be changed further down in each time step
    auto bc_fun = std::make_shared<dolfinx::fem::Constant<T>>(
        std::vector<T>({0.0, 0.0, 0.0}));
    auto bcs = {std::make_shared<const dolfinx::fem::DirichletBC<T>>(
                    bc_fun, bdofs_1, V),
                std::make_shared<const dolfinx::fem::DirichletBC<T>>(
                    std::vector<T>({0.0, 0.0, 0.0}), bdofs_2, V)};

    // create "non-linear" meshtie problem (linear problem written as non-linear
    // problem)
    auto problem
        = ThermoElasticProblem(F, J, bcs, meshties, domain1, {1, 2}, u, T0,
                               lmbda, mu, alpha_c, E * gamma, theta);
    // petsc vector corresponding to displacement function
    dolfinx::la::petsc::Vector _u(
        dolfinx::la::petsc::create_vector_wrap(*u->x()), false);

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

    // Create output file
    u->name = "displacement";
    T0->name = "temperature";
    dolfinx::io::VTXWriter<U> outfile(mesh->comm(), "results.bp", {u, T0},
                                      "BP4");
    outfile.write(0.0);

    // time step loop
    std::size_t time_steps = 80;
    for (std::size_t k = 0; k < time_steps; ++k)
    {
      //  Assemble linear thermal problem

      // Assemble vector
      b_therm.set(0.0);
      meshties->assemble_vector(b_therm.mutable_array(), *Q,
                                dolfinx_contact::Problem::Poisson);
      dolfinx::fem::assemble_vector(b_therm.mutable_array(), *L_therm);
      dolfinx::fem::apply_lifting<T, U>(b_therm.mutable_array(), {a_therm},
                                        {{bcs_therm}}, {}, double(1.0));
      b_therm.scatter_rev(std::plus<T>());
      dolfinx::fem::set_bc<T, U>(b_therm.mutable_array(), {bcs_therm});

      // Assemble matrix
      MatZeroEntries(A_therm.mat());
      meshties->assemble_matrix(
          la::petsc::Matrix::set_block_fn(A_therm.mat(), ADD_VALUES),
          *a_therm->function_spaces()[0], dolfinx_contact::Problem::Poisson);
      MatAssemblyBegin(A_therm.mat(), MAT_FLUSH_ASSEMBLY);
      MatAssemblyEnd(A_therm.mat(), MAT_FLUSH_ASSEMBLY);
      dolfinx::fem::assemble_matrix(
          dolfinx::la::petsc::Matrix::set_block_fn(A_therm.mat(), ADD_VALUES),
          *a_therm, {bcs_therm});
      MatAssemblyBegin(A_therm.mat(), MAT_FLUSH_ASSEMBLY);
      MatAssemblyEnd(A_therm.mat(), MAT_FLUSH_ASSEMBLY);

      dolfinx::fem::set_diagonal<T>(
          dolfinx::la::petsc::Matrix::set_fn(A_therm.mat(), INSERT_VALUES), *Q,
          {bcs_therm});
      MatAssemblyBegin(A_therm.mat(), MAT_FINAL_ASSEMBLY);
      MatAssemblyEnd(A_therm.mat(), MAT_FINAL_ASSEMBLY);

      // solve linear system
      ksp_therm.solve(_T.vec(), _b_therm.vec());
      // Update ghost values before output
      T0->x()->scatter_fwd();

      bc_fun->value[1] = -0.5 * (k + 1) / time_steps;
      // solve non-linear problem
      newton_solver.solve(_u.vec());
      outfile.write(k + 1);
    }

    // outfile.close();
  }
  PetscFinalize();
  return 0;
}