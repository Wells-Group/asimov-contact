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
#include <dolfinx_contact/parallel_mesh_ghosting.h>
#include <dolfinx_contact/utils.h>

using T = PetscScalar;
using U = typename dolfinx::scalar_value_type_t<T>;

int main(int argc, char* argv[])
{

  init_logging(argc, argv);
  PetscInitialize(&argc, &argv, nullptr, nullptr);

  // Set the logging thread name to show the process rank
  int mpi_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  std::string fmt = "[%Y-%m-%d %H:%M:%S.%e] [RANK " + std::to_string(mpi_rank)
                    + "] [%l] %v";
  spdlog::set_pattern(fmt);
  {
    auto [mesh_init, domain1_init, facet1_init] = dolfinx_contact::read_mesh(
        "box_3D.xdmf", "mesh", "mesh", "cell_marker", "facet_marker");

    const std::int32_t contact_bdry_1 = 6;  // top contact interface
    const std::int32_t contact_bdry_2 = 13; // bottom contact interface
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
            _f.push_back(lmbda_val);
          return {_f, {_f.size()}};
        });

    // create DG0 function for lame parameter mu
    auto mu = std::make_shared<fem::Function<T>>(V0);
    mu->interpolate(
        [mu_val](auto x) -> std::pair<std::vector<T>, std::vector<std::size_t>>
        {
          std::vector<T> _f;
          for (std::size_t p = 0; p < x.extent(1); ++p)
            _f.push_back(mu_val);
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
          MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
              double, MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 2>>
              _f(fdata.data(), bs, x.extent(1));
          for (std::size_t p = 0; p < x.extent(1); ++p)
            _f(1, p) = 0.5;
          return {std::move(fdata), {bs, x.extent(1)}};
        });

    // Function for surface traction
    auto t = std::make_shared<fem::Function<T>>(V);

    t->interpolate(
        [bs](auto x) -> std::pair<std::vector<T>, std::vector<std::size_t>>
        {
          std::vector<T> fdata(bs * x.extent(1), 0.0);
          namespace stdex = std::experimental;
          MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
              double, MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 2>>
              _f(fdata.data(), bs, x.extent(1));
          for (std::size_t p = 0; p < x.extent(1); ++p)
            _f(1, p) = 0.5;
          return {std::move(fdata), {bs, x.extent(1)}};
        });

    // Create integration domains for integrating over specific surfaces
    std::vector<std::pair<std::int32_t, std::span<const std::int32_t>>>
        integration_domain;
    std::vector<std::vector<std::int32_t>> facet_domains;
    {
      // Get unique values in facet1 MeshTags
      std::vector ids(facet1.values().begin(), facet1.values().end());
      std::sort(ids.begin(), ids.end());
      ids.erase(std::unique(ids.begin(), ids.end()), ids.end());

      // Pack (domain id, indices) pairs
      for (auto id : ids)
      {
        facet_domains.push_back(fem::compute_integration_domains(
            fem::IntegralType::exterior_facet, *facet1.topology(),
            facet1.find(id), facet1.dim()));
        integration_domain.emplace_back(id, facet_domains.back());
      }
    }

    // Define variational forms
    auto J = std::make_shared<fem::Form<T>>(
        fem::create_form<T>(*form_linear_elasticity_J, {V, V},
                            {{"mu", mu}, {"lmbda", lmbda}}, {}, {}));
    auto F = std::make_shared<fem::Form<T>>(fem::create_form<T>(
        *form_linear_elasticity_F, {V}, {{"f", f}, {"t", t}}, {},
        {{dolfinx::fem::IntegralType::exterior_facet, integration_domain}}));

    // Define boundary conditions
    const std::int32_t dirichlet_bdy = 12; // bottom face
    auto facets = facet1.find(dirichlet_bdy);
    auto bdofs = dolfinx::fem::locate_dofs_topological(
        *V->mesh()->topology_mutable(), *V->dofmap(), 2, facets);
    auto bc = std::make_shared<const dolfinx::fem::DirichletBC<T>>(
        std::vector<T>({0.0, 0.0, 0.0}), bdofs, V);

    // Create meshties
    std::vector<std::int32_t> data = {contact_bdry_1, contact_bdry_2};
    std::vector<std::int32_t> offsets = {0, 2};
    auto contact_markers
        = std::make_shared<dolfinx::graph::AdjacencyList<std::int32_t>>(
            std::move(data), std::move(offsets));
    std::vector<std::shared_ptr<dolfinx::mesh::MeshTags<std::int32_t>>> markers
        = {std::make_shared<dolfinx::mesh::MeshTags<std::int32_t>>(facet1)};
    std::vector<std::array<int, 2>> pairs = {{0, 1}, {1, 0}};
    auto meshties
        = dolfinx_contact::MeshTie(markers, contact_markers, pairs, mesh, 5);

    meshties.generate_kernel_data(dolfinx_contact::Problem::Elasticity, V,
                                  {{"mu", mu}, {"lambda", lmbda}}, E * gamma,
                                  theta);

    // Create matrix and vector
    auto A = dolfinx::la::petsc::Matrix(
        meshties.create_petsc_matrix(*J, std::string()), false);
    dolfinx::la::Vector<T> b(F->function_spaces()[0]->dofmap()->index_map,
                             F->function_spaces()[0]->dofmap()->index_map_bs());

    // Assemble vector
    b.set(0.0);
    meshties.assemble_vector(b.mutable_array(), V,
                             dolfinx_contact::Problem::Elasticity);
    dolfinx::fem::assemble_vector(b.mutable_array(), *F);
    dolfinx::fem::apply_lifting<T, U>(b.mutable_array(), {J}, {{bc}}, {},
                                      double(1.0));
    b.scatter_rev(std::plus<T>());
    dolfinx::fem::set_bc<T, U>(b.mutable_array(), {bc});

    // Assemble matrix
    MatZeroEntries(A.mat());
    meshties.assemble_matrix(
        la::petsc::Matrix::set_block_fn(A.mat(), ADD_VALUES),
        J->function_spaces()[0], dolfinx_contact::Problem::Elasticity);
    MatAssemblyBegin(A.mat(), MAT_FLUSH_ASSEMBLY);
    MatAssemblyEnd(A.mat(), MAT_FLUSH_ASSEMBLY);
    dolfinx::fem::assemble_matrix(
        dolfinx::la::petsc::Matrix::set_block_fn(A.mat(), ADD_VALUES), *J,
        {bc});
    MatAssemblyBegin(A.mat(), MAT_FLUSH_ASSEMBLY);
    MatAssemblyEnd(A.mat(), MAT_FLUSH_ASSEMBLY);

    dolfinx::fem::set_diagonal<T>(
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
    dolfinx::la::petsc::KrylovSolver ksp(MPI_COMM_WORLD);
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

    ksp.set_from_options();
    ksp.set_operator(A.mat());

    // displacement function
    auto u = std::make_shared<fem::Function<T>>(V);
    dolfinx::la::petsc::Vector _u(
        dolfinx::la::petsc::create_vector_wrap(*u->x()), false);
    dolfinx::la::petsc::Vector _b(dolfinx::la::petsc::create_vector_wrap(b),
                                  false);

    // solve linear system
    ksp.solve(_u.vec(), _b.vec());

    // Update ghost values before output
    u->x()->scatter_fwd();
    // loguru::g_stderr_verbosity = loguru::Verbosity_OFF;
    dolfinx::io::XDMFFile file2(mesh->comm(), "result.xdmf", "w");
    file2.write_mesh(*mesh);
    file2.write_function(*u, 0.0);
    file2.close();
  }
  PetscFinalize();
  return 0;
}