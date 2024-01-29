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
//----------------------------------------------------------------------------
/// Read a mesh
/// @param[in] filename The file name
/// @return The tuple (mesh, domain tags, facet tags)
//----------------------------------------------------------------------------
auto read_mesh(const std::string& filename)
{
  // Read and create mesh
  dolfinx::io::XDMFFile file(MPI_COMM_WORLD, filename, "r");
  dolfinx::fem::CoordinateElement<double> cmap
      = dolfinx::fem::CoordinateElement<double>(
          dolfinx::mesh::CellType::tetrahedron, 1);

  auto [x, xshape] = file.read_geometry_data("mesh");
  auto [cells, cshape] = file.read_topology_data("mesh");
  std::vector<std::int32_t> offset(cshape[0] + 1, 0);
  for (std::size_t i = 0; i < cshape[0]; ++i)
    offset[i + 1] = offset[i] + cshape[1];

  const std::vector<double>& _x = std::get<std::vector<double>>(x);
  auto mesh = std::make_shared<dolfinx::mesh::Mesh<double>>(
      dolfinx::mesh::create_mesh(MPI_COMM_WORLD,
                                 std::span<const std::int64_t>(cells), {cmap},
                                 _x, xshape, dolfinx::mesh::GhostMode::none));

  mesh->topology_mutable()->create_entities(2);
  mesh->topology_mutable()->create_connectivity(2, 3);

  // Create entity-vertex connectivity
  constexpr int tdim = 3;
  mesh->topology_mutable()->create_entities(tdim - 1);
  mesh->topology_mutable()->create_connectivity(tdim - 1, tdim);

  // Read domain meshtags
  if (dolfinx::MPI::rank(mesh->comm()) == 0)
    std::cout << "Reading domain MeshTags ..." << std::endl;
  auto domain1 = file.read_meshtags(*mesh, "cell_marker");

  // Read facet meshtags
  if (dolfinx::MPI::rank(mesh->comm()) == 0)
    std::cout << "Reading facet MeshTags ..." << std::endl;
  auto facet1 = file.read_meshtags(*mesh, "facet_marker");

  file.close();

  return std::make_tuple(mesh, domain1, facet1);
}

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
    auto [mesh, domain1, facet1] = read_mesh("box_3D.xdmf");
    // Create function spaces
    auto V = std::make_shared<fem::FunctionSpace<double>>(
        fem::create_functionspace(functionspace_form_linear_elasticity_J, "w",
                                  mesh));
    auto V0 = std::make_shared<fem::FunctionSpace<double>>(
        fem::create_functionspace(functionspace_form_linear_elasticity_J, "mu",
                                  mesh));

    double E = 1000;
    double nu = 0.1;
    double gamma = 10;
    double theta = 1;

    double lmbda_val = E * nu / ((1 + nu) * (1 - nu));
    double mu_val = E / (2 * (1 + nu));

    // Create DG0 function for lame parameter lambda
    auto lmbda = std::make_shared<fem::Function<double>>(V0);
    lmbda->interpolate(
        [lmbda_val](
            auto x) -> std::pair<std::vector<double>, std::vector<std::size_t>>
        {
          std::vector<double> _f;
          for (std::size_t p = 0; p < x.extent(1); ++p)
          {
            _f.push_back(lmbda_val);
          }
          return {_f, {_f.size()}};
        });

    // create DG0 function for lame parameter mu
    auto mu = std::make_shared<fem::Function<double>>(V0);
    mu->interpolate(
        [mu_val](
            auto x) -> std::pair<std::vector<double>, std::vector<std::size_t>>
        {
          std::vector<double> _f;
          for (std::size_t p = 0; p < x.extent(1); ++p)
          {
            _f.push_back(mu_val);
          }
          return {_f, {_f.size()}};
        });

    // Function for body force
    auto f = std::make_shared<fem::Function<double>>(V);
    std::size_t bs = V->dofmap()->bs();
    f->interpolate(
        [bs](auto x) -> std::pair<std::vector<double>, std::vector<std::size_t>>
        {
          std::vector<double> fdata(bs * x.extent(1), 0.0);
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
    auto t = std::make_shared<fem::Function<double>>(V);

    t->interpolate(
        [bs](auto x) -> std::pair<std::vector<double>, std::vector<std::size_t>>
        {
          std::vector<double> fdata(bs * x.extent(1), 0.0);
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
    auto J = std::make_shared<fem::Form<double>>(
        fem::create_form<double>(*form_linear_elasticity_J, {V, V},
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
    auto F = std::make_shared<fem::Form<double>>(fem::create_form<double>(
        *form_linear_elasticity_F, {V}, {{"f", f}, {"t", t}}, {},
        {{dolfinx::fem::IntegralType::exterior_facet, integration_domain}}));

    // Define boundary conditions
    const std::int32_t dirichlet_bdy = 12; // bottom face
    auto facets = facet1.find(dirichlet_bdy);
    auto bdofs = dolfinx::fem::locate_dofs_topological(
        *V->mesh()->topology_mutable(), *V->dofmap(), 2, facets);
    auto bc = std::make_shared<const dolfinx::fem::DirichletBC<double>>(
        std::vector<double>({0.0, 0.0, 0.0}), bdofs, V);

    // Create meshties
    const std::int32_t contact_bdy_1 = 6;  // top contact interface
    const std::int32_t contact_bdy_2 = 13; // bottom contact interface
    std::vector<std::int32_t> data = {contact_bdy_1, contact_bdy_2};
    std::vector<std::int32_t> offsets = {0, 2};
    auto contact_markers
        = std::make_shared<dolfinx::graph::AdjacencyList<std::int32_t>>(
            std::move(data), std::move(offsets));
    std::vector<std::shared_ptr<dolfinx::mesh::MeshTags<std::int32_t>>> markers
        = {std::make_shared<dolfinx::mesh::MeshTags<std::int32_t>>(facet1)};
    std::vector<std::array<int, 2>> pairs = {{0, 1}, {1, 0}};
    auto meshties
        = dolfinx_contact::MeshTie(markers, contact_markers, pairs, V, 5);

    meshties.generate_meshtie_data_matrix_only(lmbda, mu, E * gamma, theta);

    // Create matrix and vector
    auto A = dolfinx::la::petsc::Matrix(
        meshties.create_petsc_matrix(*J, std::string()), false);
    dolfinx::la::Vector<double> b(
        F->function_spaces()[0]->dofmap()->index_map,
        F->function_spaces()[0]->dofmap()->index_map_bs());

    // Assemble vector
    b.set(0.0);
    meshties.assemble_vector(b.mutable_array());
    dolfinx::fem::assemble_vector(b.mutable_array(), *F);
    dolfinx::fem::apply_lifting<double, double>(b.mutable_array(), {J}, {{bc}},
                                                {}, double(1.0));
    b.scatter_rev(std::plus<double>());
    dolfinx::fem::set_bc<double, double>(b.mutable_array(), {bc});

    // Assemble matrix
    MatZeroEntries(A.mat());
    meshties.assemble_matrix(
        la::petsc::Matrix::set_block_fn(A.mat(), ADD_VALUES));
    MatAssemblyBegin(A.mat(), MAT_FLUSH_ASSEMBLY);
    MatAssemblyEnd(A.mat(), MAT_FLUSH_ASSEMBLY);
    dolfinx::fem::assemble_matrix(
        dolfinx::la::petsc::Matrix::set_block_fn(A.mat(), ADD_VALUES), *J,
        {bc});
    MatAssemblyBegin(A.mat(), MAT_FLUSH_ASSEMBLY);
    MatAssemblyEnd(A.mat(), MAT_FLUSH_ASSEMBLY);

    dolfinx::fem::set_diagonal<double>(
        dolfinx::la::petsc::Matrix::set_fn(A.mat(), INSERT_VALUES), *V, {bc});
    MatAssemblyBegin(A.mat(), MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(A.mat(), MAT_FINAL_ASSEMBLY);

    // Build near-nullspace and attach to matrix
    std::vector<std::int32_t> tags = {1, 2};
    MatNullSpace ns
        = dolfinx_contact::build_nullspace_multibody(*V, domain1, tags);
    MatSetNearNullSpace(A.mat(), ns);
    MatNullSpaceDestroy(&ns);

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
    dolfinx::la::petsc::options::set("ksp_monitor");

    lu.set_from_options();
    lu.set_operator(A.mat());

    // displacement function
    auto u = std::make_shared<fem::Function<double>>(V);
    dolfinx::la::petsc::Vector _u(
        dolfinx::la::petsc::create_vector_wrap(*u->x()), false);
    dolfinx::la::petsc::Vector _b(dolfinx::la::petsc::create_vector_wrap(b),
                                  false);

    // solve linear system
    lu.solve(_u.vec(), _b.vec());

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