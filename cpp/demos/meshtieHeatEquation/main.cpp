// Copyright (C) 2023 Sarah Roggendorf
//
// This file is part of DOLFINx_Contact
//
// SPDX-License-Identifier:    MIT

// meshtie demo for the heat equation with implicit Euler
// ====================================================

#include "heat_equation.h"
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
    auto [mesh_init, domain1_init, facet1_init]
        = dolfinx_contact::read_mesh("../meshes/cont-blocks_sk24_fnx.xdmf");

    const std::int32_t contact_bdry_1 = 6;  // top contact interface
    const std::int32_t contact_bdry_2 = 12; // bottom contact interface
    // loguru::g_stderr_verbosity = loguru::Verbosity_OFF;
    auto [mesh_new, facet1, domain1] = dolfinx_contact::create_contact_mesh(
        *mesh_init, facet1_init, domain1_init, {contact_bdry_1, contact_bdry_2},
        10.0);
    auto mesh = std::make_shared<dolfinx::mesh::Mesh<U>>(mesh_new);

    // Create function spaces
    auto ct = mesh->topology()->cell_type();
    auto element_DG = basix::create_element<double>(
        basix::element::family::P, dolfinx::mesh::cell_type_to_basix_type(ct),
        0, basix::element::lagrange_variant::unset,
        basix::element::dpc_variant::unset, true);
    auto element = basix::create_element<double>(
        basix::element::family::P, dolfinx::mesh::cell_type_to_basix_type(ct),
        1, basix::element::lagrange_variant::unset,
        basix::element::dpc_variant::unset, false);
    auto Q = std::make_shared<fem::FunctionSpace<U>>(
        fem::create_functionspace(mesh, element));
    auto V0 = std::make_shared<fem::FunctionSpace<U>>(
        fem::create_functionspace(mesh, element_DG));

    // Nitsche parameters
    double gamma = 10;
    double theta = 1;

    // Create DG0 function for time-step/heat coefficient
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

    // Temperature function
    auto T0 = std::make_shared<fem::Function<T>>(Q);

    // Define variational forms
    auto a_therm = std::make_shared<fem::Form<T>>(
        fem::create_form<T>(*form_heat_equation_a_therm, {Q, Q},
                            {{"T0", T0}, {"kdt", kdt}}, {}, {}));
    auto L_therm = std::make_shared<fem::Form<T>>(fem::create_form<T>(
        *form_heat_equation_L_therm, {Q}, {{"T0", T0}}, {}, {}));

    // Define boundary conditions
    const std::int32_t dirichlet_bdy = 2; // bottom face
    auto facets = facet1.find(dirichlet_bdy);
    auto bdofs = dolfinx::fem::locate_dofs_topological(
        *Q->mesh()->topology_mutable(), *Q->dofmap(), 2, facets);
    auto bc
        = std::make_shared<const dolfinx::fem::DirichletBC<T>>(1.0, bdofs, Q);

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

    meshties.generate_kernel_data(dolfinx_contact::Problem::Poisson, Q,
                                  {{"kdt", kdt}}, gamma, theta);

    // Create matrix and vector
    auto A_therm = dolfinx::la::petsc::Matrix(
        meshties.create_petsc_matrix(*a_therm, std::string()), false);
    dolfinx::la::Vector<T> b_therm(
        L_therm->function_spaces()[0]->dofmap()->index_map,
        L_therm->function_spaces()[0]->dofmap()->index_map_bs());

    // Set up linear solver with parameters
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

    // petsc vectors for temperature and rhs
    dolfinx::la::petsc::Vector _T(
        dolfinx::la::petsc::create_vector_wrap(*T0->x()), false);
    dolfinx::la::petsc::Vector _b_therm(
        dolfinx::la::petsc::create_vector_wrap(b_therm), false);

    // set up output file
    dolfinx::io::VTXWriter<U> outfile(mesh->comm(), "results.bp", {T0}, "BP4");
    outfile.write(0.0);

    // time stepping loop
    std::size_t time_steps = 40;
    for (std::size_t k = 0; k < time_steps; ++k)
    {

      // Assemble vector
      b_therm.set(0.0);
      meshties.assemble_vector(b_therm.mutable_array(), Q,
                               dolfinx_contact::Problem::Poisson);
      dolfinx::fem::assemble_vector(b_therm.mutable_array(), *L_therm);
      dolfinx::fem::apply_lifting<T, U>(b_therm.mutable_array(), {a_therm},
                                        {{bc}}, {}, double(1.0));
      b_therm.scatter_rev(std::plus<T>());
      dolfinx::fem::set_bc<T, U>(b_therm.mutable_array(), {bc});

      // Assemble matrix
      MatZeroEntries(A_therm.mat());
      meshties.assemble_matrix(
          la::petsc::Matrix::set_block_fn(A_therm.mat(), ADD_VALUES),
          a_therm->function_spaces()[0], dolfinx_contact::Problem::Poisson);
      MatAssemblyBegin(A_therm.mat(), MAT_FLUSH_ASSEMBLY);
      MatAssemblyEnd(A_therm.mat(), MAT_FLUSH_ASSEMBLY);
      dolfinx::fem::assemble_matrix(
          dolfinx::la::petsc::Matrix::set_block_fn(A_therm.mat(), ADD_VALUES),
          *a_therm, {bc});
      MatAssemblyBegin(A_therm.mat(), MAT_FLUSH_ASSEMBLY);
      MatAssemblyEnd(A_therm.mat(), MAT_FLUSH_ASSEMBLY);

      dolfinx::fem::set_diagonal<T>(
          dolfinx::la::petsc::Matrix::set_fn(A_therm.mat(), INSERT_VALUES), *Q,
          {bc});
      MatAssemblyBegin(A_therm.mat(), MAT_FINAL_ASSEMBLY);
      MatAssemblyEnd(A_therm.mat(), MAT_FINAL_ASSEMBLY);

      // solve linear system
      ksp_therm.solve(_T.vec(), _b_therm.vec());

      // Update ghost values before output
      T0->x()->scatter_fwd();
      outfile.write(double(k + 1));
    }
    outfile.close();
  }
  PetscFinalize();
  return 0;
}