// Copyright (C) 2023 Sarah Roggendorf
//
// This file is part of DOLFINx_Contact
//
// SPDX-License-Identifier:    MIT

// meshtie demo
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
#include <dolfinx_contact/utils.h>

using T = PetscScalar;
using U = typename dolfinx::scalar_value_type_t<T>;
//----------------------------------------------------------------------------
/// Read a mesh
/// @param[in] filename The file name
/// @return The tuple (mesh, domain tags, facet tags)
//----------------------------------------------------------------------------
auto read_mesh(const std::string& filename,
               const std::string& topo_name = "volume markers",
               const std::string& geo_name = "geometry",
               const std::string& volume_markers = "volume markers",
               const std::string& facet_markers = "facet markers")
{
  // Read and create mesh
  dolfinx::io::XDMFFile file(MPI_COMM_WORLD, filename, "r");
  auto [ct, cdegree] = file.read_cell_type(volume_markers);
  dolfinx::fem::CoordinateElement<U> cmap
      = dolfinx::fem::CoordinateElement<U>(ct, cdegree);

  auto [x, xshape] = file.read_geometry_data(geo_name);
  auto [cells, cshape] = file.read_topology_data(topo_name);
  std::vector<std::int32_t> offset(cshape[0] + 1, 0);
  for (std::size_t i = 0; i < cshape[0]; ++i)
    offset[i + 1] = offset[i] + cshape[1];

  dolfinx::graph::AdjacencyList<std::int64_t> cells_adj(std::move(cells),
                                                        std::move(offset));
  const std::vector<U>& _x = std::get<std::vector<U>>(x);
  auto mesh = std::make_shared<dolfinx::mesh::Mesh<U>>(
      dolfinx::mesh::create_mesh(MPI_COMM_WORLD, cells_adj, {cmap}, _x, xshape,
                                 dolfinx::mesh::GhostMode::none));

  mesh->topology_mutable()->create_entities(2);
  mesh->topology_mutable()->create_connectivity(2, 3);

  // Create entity-vertex connectivity
  constexpr int tdim = 3;
  mesh->topology_mutable()->create_entities(tdim - 1);
  mesh->topology_mutable()->create_connectivity(tdim - 1, tdim);

  // Read domain meshtags
  if (dolfinx::MPI::rank(mesh->comm()) == 0)
    std::cout << "Reading domain MeshTags ..." << std::endl;
  auto domain1 = file.read_meshtags(*mesh, volume_markers);

  // Read facet meshtags
  if (dolfinx::MPI::rank(mesh->comm()) == 0)
    std::cout << "Reading facet MeshTags ..." << std::endl;
  auto facet1 = file.read_meshtags(*mesh, facet_markers);

  file.close();

  return std::make_tuple(mesh, domain1, facet1);
}

class MeshTieProblem
{
public:
  MeshTieProblem(
      std::shared_ptr<dolfinx::fem::Form<T>> L,
      std::shared_ptr<dolfinx::fem::Form<T>> J,
      std::vector<std::shared_ptr<const dolfinx::fem::DirichletBC<T>>> bcs,
      std::shared_ptr<dolfinx_contact::MeshTie> meshties,
      dolfinx::mesh::MeshTags<std::int32_t> subdomains,
      std::shared_ptr<dolfinx::fem::Function<T>> lmbda,
      std::shared_ptr<dolfinx::fem::Function<T>> mu, double gamma, double theta)
      : _l(L), _j(J), _bcs(bcs), _meshties(meshties),
        _b(L->function_spaces()[0]->dofmap()->index_map,
           L->function_spaces()[0]->dofmap()->index_map_bs()),
        _matA(dolfinx::la::petsc::Matrix(
            meshties->create_petsc_matrix(*J, std::string()), false)),
        _lmbda(lmbda), _mu(mu), _gamma(gamma), _theta(theta)
  {
      
    auto map = L->function_spaces()[0]->dofmap()->index_map;
    const int bs = L->function_spaces()[0]->dofmap()->index_map_bs();
    std::int32_t size_local = bs * map->size_local();

    std::vector<PetscInt> ghosts(map->ghosts().begin(), map->ghosts().end());
    std::int64_t size_global = bs * map->size_global();
    VecCreateGhostBlockWithArray(map->comm(), bs, size_local, size_global,
                                 ghosts.size(), ghosts.data(),
                                 _b.array().data(), &_b_petsc);
 
    _meshties->generate_meshtie_data_matrix_only(lmbda, mu, gamma, theta);
    // Build near-nullspace and attach to matrix

    std::vector<std::int32_t> tags;
    tags.assign(subdomains.values().begin(), subdomains.values().end());
    tags.erase(std::unique(tags.begin(), tags.end()),
                     tags.end());        
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

  /// Compute F at current point x
  auto F()
  {
    return [&](const Vec x, Vec)
    {
      // Assemble b
      std::span<T> b(_b.mutable_array());
      std::fill(b.begin(), b.end(), 0.0);
      _meshties->assemble_vector(b);
      fem::assemble_vector<T>(b, *_l);
      
      //Apply lifting
      dolfinx::fem::apply_lifting<T, U>(b, {_j}, {_bcs},
                                                {}, T(1.0));
      _b.scatter_rev(std::plus<T>());

      // Set bc
      Vec x_local;
      VecGhostGetLocalForm(x, &x_local);
      PetscInt n = 0;
      VecGetSize(x_local, &n);
      const T* array = nullptr;
      VecGetArrayRead(x_local, &array);
      fem::set_bc<T>(b, _bcs, std::span<const T>(array, n), 1.0);
      VecRestoreArrayRead(x, &array);
    };
  }

  auto J()
  {
    return [&](const Vec, Mat A)
    {
      // Assemble matrix
      MatZeroEntries(A);
      _meshties->assemble_matrix(
          la::petsc::Matrix::set_block_fn(A, ADD_VALUES));
      MatAssemblyBegin(A, MAT_FLUSH_ASSEMBLY);
      MatAssemblyEnd(A, MAT_FLUSH_ASSEMBLY);
      dolfinx::fem::assemble_matrix(
          dolfinx::la::petsc::Matrix::set_block_fn(A, ADD_VALUES), *_j,
          _bcs);
      MatAssemblyBegin(A, MAT_FLUSH_ASSEMBLY);
      MatAssemblyEnd(A, MAT_FLUSH_ASSEMBLY);

      dolfinx::fem::set_diagonal<T>(
          dolfinx::la::petsc::Matrix::set_fn(A, INSERT_VALUES),
          *_j->function_spaces()[0], _bcs);
      MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
      MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);
    };
  }

  Vec vector() { return _b_petsc; }

  Mat matrix() { return _matA.mat(); }

private:
  std::shared_ptr<dolfinx::fem::Form<T>> _l, _j;
  std::vector<std::shared_ptr<const dolfinx::fem::DirichletBC<T>>> _bcs;
  std::shared_ptr<dolfinx_contact::MeshTie> _meshties;
  dolfinx::la::Vector<T> _b;
  Vec _b_petsc = nullptr;
  la::petsc::Matrix _matA;
  std::shared_ptr<dolfinx::fem::Function<T>> _lmbda;
  std::shared_ptr<dolfinx::fem::Function<T>> _mu;
  double _gamma;
  double _theta;
  
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
    auto [mesh, domain1, facet1] = read_mesh("box_3D.xdmf", "mesh", "mesh",
                                             "cell_marker", "facet_marker");
    // Create function spaces
    auto V = std::make_shared<fem::FunctionSpace<U>>(fem::create_functionspace(
        functionspace_form_linear_elasticity_J, "w", mesh));
    auto V0 = std::make_shared<fem::FunctionSpace<U>>(fem::create_functionspace(
        functionspace_form_linear_elasticity_J, "mu", mesh));

    double E = 1000;
    double nu = 0.1;
    double gamma = 10;
    double theta = 1;

    double lmbda_val = E * nu / ((1 + nu) * (1 - nu));
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

    // Function for body force
    auto f = std::make_shared<fem::Function<T>>(V);
    std::size_t bs = V->dofmap()->bs();
    f->interpolate(
        [bs](auto x) -> std::pair<std::vector<T>, std::vector<std::size_t>>
        {
          std::vector<T> fdata(bs * x.extent(1), 0.0);
          namespace stdex = std::experimental;
          stdex::mdspan<
              double, MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 2>>
              _f(fdata.data(), bs, x.extent(1));
          for (std::size_t p = 0; p < x.extent(1); ++p)
          {
            _f(1, p) = 0.5;
          }
          return {std::move(fdata), {bs, x.extent(1)}};
        });

    // Function for surface traction
    auto t = std::make_shared<fem::Function<T>>(V);

    t->interpolate(
        [bs](auto x) -> std::pair<std::vector<T>, std::vector<std::size_t>>
        {
          std::vector<T> fdata(bs * x.extent(1), 0.0);
          namespace stdex = std::experimental;
          stdex::mdspan<
              double, MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 2>>
              _f(fdata.data(), bs, x.extent(1));
          for (std::size_t p = 0; p < x.extent(1); ++p)
          {
            _f(1, p) = 0.5;
          }
          return {std::move(fdata), {bs, x.extent(1)}};
        });

    // create integration domains for integrating over specific surfaces
    auto facet_domains = fem::compute_integration_domains(
        fem::IntegralType::exterior_facet, *facet1.topology(), facet1.indices(),
        facet1.dim(), facet1.values());

    // Define variational forms
    auto J = std::make_shared<fem::Form<T>>(
        fem::create_form<T>(*form_linear_elasticity_J, {V, V},
                            {{"mu", mu}, {"lmbda", lmbda}}, {}, {}));
    auto F = std::make_shared<fem::Form<T>>(fem::create_form<T>(
        *form_linear_elasticity_F, {V}, {{"f", f}, {"t", t}}, {},
        {{dolfinx::fem::IntegralType::exterior_facet, facet_domains}}));

    // Define boundary conditions
    const std::int32_t dirichlet_bdy = 12; // bottom face
    auto facets = facet1.find(dirichlet_bdy);
    auto bdofs = dolfinx::fem::locate_dofs_topological(
        *V->mesh()->topology_mutable(), *V->dofmap(), 2, facets);
    auto bc = std::make_shared<const dolfinx::fem::DirichletBC<T>>(
        std::vector<T>({0.0, 0.0, 0.0}), bdofs, V);

    // Create meshties
    const std::int32_t contact_bdry_1 = 6;  // top contact interface
    const std::int32_t contact_bdry_2 = 13; // bottom contact interface
    std::vector<std::int32_t> data = {contact_bdry_1, contact_bdry_2};
    std::vector<std::int32_t> offsets = {0, 2};
    auto contact_markers
        = std::make_shared<dolfinx::graph::AdjacencyList<std::int32_t>>(
            std::move(data), std::move(offsets));
    std::vector<std::shared_ptr<dolfinx::mesh::MeshTags<std::int32_t>>> markers
        = {std::make_shared<dolfinx::mesh::MeshTags<std::int32_t>>(facet1)};
    std::vector<std::array<int, 2>> pairs = {{0, 1}, {1, 0}};
    auto meshties = std::make_shared<dolfinx_contact::MeshTie>(
        dolfinx_contact::MeshTie(markers, contact_markers, pairs, V, 5));

    auto problem = MeshTieProblem(F, J, {bc}, meshties, domain1, lmbda, mu, gamma * E,
                             theta);               

    // Set up linear solver with parameters
    dolfinx::la::petsc::KrylovSolver lu(MPI_COMM_WORLD);
    dolfinx::la::petsc::options::set("ksp_type", "cg");
    dolfinx::la::petsc::options::set("pc_type", "gamg");
    dolfinx::la::petsc::options::set("pc_mg_levels", 3);
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

    lu.set_from_options();
    lu.set_operator(problem.matrix());
      
    // displacement function
    auto u = std::make_shared<fem::Function<T>>(V);
    dolfinx::la::petsc::Vector _u(
        dolfinx::la::petsc::create_vector_wrap(*u->x()), false);
    auto assemble_vec = problem.F();
    auto assemble_mat = problem.J();   
    assemble_vec(_u.vec(), problem.vector());
    assemble_mat(_u.vec(), problem.matrix());
       
    // solve linear system
    lu.solve(_u.vec(), problem.vector());

    // Update ghost values before output
    u->x()->scatter_fwd();
    dolfinx::io::XDMFFile file2(mesh->comm(), "result.xdmf", "w");
    file2.write_mesh(*mesh);
    file2.write_function(*u, 0.0);
    file2.close();
  }
  PetscFinalize();
  return 0;
}