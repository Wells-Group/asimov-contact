//
// testing of large number of meshties
// written by Neeraj Cherukunnath
// two dirichlet faces, domains with CFload 
// 
#include "linear_elasticity.h"
// DOLFINx
#include <dolfinx.h>
#include <dolfinx/common/log.h>
#include <dolfinx/common/sort.h>
#include <dolfinx/fem/Constant.h>
#include <dolfinx/fem/Expression.h>
#include <dolfinx/fem/petsc.h>
#include <dolfinx/fem/assembler.h>
#include <dolfinx/io/XDMFFile.h>
#include <dolfinx/io/VTKFile.h>
#include <dolfinx/io/cells.h>
#include <dolfinx/io/xdmf_utils.h>
#include <dolfinx/la/Vector.h>
#include <dolfinx/la/petsc.h>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/mesh/cell_types.h>
#include <dolfinx/nls/NewtonSolver.h>

// DOLFINx_CONTACT
#include <dolfinx_contact/Contact.h>
#include <dolfinx_contact/MeshTie.h>
#include <dolfinx_contact/parallel_mesh_ghosting.h>
#include <dolfinx_contact/utils.h>

// Basix
#include <basix/finite-element.h>
#include <basix/e-lagrange.h>

// Other headers
#include <cmath>
#include <complex>
#include <filesystem>
#include <fstream>
#include <functional>
#include <math.h>
#include <string>
#include <utility>
#include <vector>

using T = PetscScalar;
using U = typename dolfinx::scalar_value_type_t<T>;

//----------------------------------------------------------------------------
/// Read a mesh
/// @param[in] filename The file name
/// @return The tuple (mesh, domain tags, facet tags)
//----------------------------------------------------------------------------
auto read_mesh(const std::string& filename)
{
  // Read and create mesh
  dolfinx::io::XDMFFile file(MPI_COMM_WORLD, filename, "r");
  dolfinx::fem::CoordinateElement cmap = dolfinx::fem::CoordinateElement<U>(
      dolfinx::mesh::CellType::tetrahedron, 1);
  auto geometry = file.read_geometry_data("geometry");
  auto topology = file.read_topology_data("volume markers");
  dolfinx::graph::AdjacencyList<std::int64_t> cells_adj
       = dolfinx::graph::regular_adjacency_list(topology.first,topology.second.back());

  MPI_Comm comm = MPI_COMM_WORLD;
  std::vector<dolfinx::fem::CoordinateElement<U>> elements = {cmap};
  auto xvector = std::get<std::vector<U>>(geometry.first);
  std::array<std::size_t, 2> xshape = geometry.second;
  auto mesh = std::make_shared<dolfinx::mesh::Mesh<U>>(
    dolfinx::mesh::create_mesh(comm, cells_adj, elements, xvector, xshape,
                                dolfinx::mesh::GhostMode::shared_facet));

  mesh->topology_mutable()->create_entities(2);
  mesh->topology_mutable()->create_connectivity(2, 3);

  // Create entity-vertex connectivity
  constexpr int tdim = 3;
  mesh->topology_mutable()->create_entities(tdim - 1);
  mesh->topology_mutable()->create_connectivity(tdim - 1, tdim);
  // Read domain meshtags
  if (dolfinx::MPI::rank(mesh->comm()) == 0)
    std::cout << "Reading domain MeshTags ..." << std::endl;
  auto domain1 = file.read_meshtags(*mesh, "volume markers");

  // Read facet meshtags
  if (dolfinx::MPI::rank(mesh->comm()) == 0)
    std::cout << "Reading facet MeshTags ..." << std::endl;
  auto facet1 = file.read_meshtags(*mesh, "facet markers");

  return std::make_tuple(mesh, domain1, facet1);
}

//----------------------------------------------------------------------------
// Function to compute the near nullspace for elasticity - it is made up
// of the six rigid body modes
//----------------------------------------------------------------------------
MatNullSpace build_near_nullspace(const fem::FunctionSpace<double>& V)
{
  // Create vectors for nullspace basis
  auto map = V.dofmap()->index_map;
  int bs = V.dofmap()->index_map_bs();
  std::vector<la::Vector<T>> basis(6, la::Vector<T>(map, bs));

  // x0, x1, x2 translations
  std::int32_t length_block = map->size_local() + map->num_ghosts();
  for (int k = 0; k < 3; ++k)
  {
    std::span<T> x = basis[k].mutable_array();
    for (std::int32_t i = 0; i < length_block; ++i)
      x[bs * i + k] = 1.0;
  }

  // Rotations
  auto x3 = basis[3].mutable_array();
  auto x4 = basis[4].mutable_array();
  auto x5 = basis[5].mutable_array();

  const std::vector<double> x = V.tabulate_dof_coordinates(false);
  const std::int32_t* dofs = V.dofmap()->map().data_handle();
  for (std::size_t i = 0; i < V.dofmap()->map().size(); ++i)
  {
    std::span<const double, 3> xd(x.data() + 3 * dofs[i], 3);

    x3[bs * dofs[i] + 0] = -xd[1];
    x3[bs * dofs[i] + 1] = xd[0];

    x4[bs * dofs[i] + 0] = xd[2];
    x4[bs * dofs[i] + 2] = -xd[0];

    x5[bs * dofs[i] + 2] = xd[1];
    x5[bs * dofs[i] + 1] = -xd[2];
  }

  // Orthonormalize basis
  la::orthonormalize(std::vector<std::reference_wrapper<la::Vector<T>>>(
      basis.begin(), basis.end()));
  if (!la::is_orthonormal(
          std::vector<std::reference_wrapper<const la::Vector<T>>>(
              basis.begin(), basis.end())))
  {
    throw std::runtime_error("Space not orthonormal");
  }

  // Build PETSc nullspace object
  std::int32_t length = bs * map->size_local();
  std::vector<std::span<const T>> basis_local;
  std::transform(basis.cbegin(), basis.cend(), std::back_inserter(basis_local),
                 [length](auto& x)
                 { return std::span(x.array().data(), length); });
  MPI_Comm comm = V.mesh()->comm();
  std::vector<Vec> v = la::petsc::create_vectors(comm, basis_local);
  MatNullSpace ns = la::petsc::create_nullspace(comm, v);
  std::for_each(v.begin(), v.end(), [](auto v) { VecDestroy(&v); });
  return ns;
}

int main(int argc, char* argv[])
{

  init_logging(argc, argv);
  PetscInitialize(&argc, &argv, nullptr, nullptr);
  // Set the logging thread name to show the process rank
  const int rank = dolfinx::MPI::rank(MPI_COMM_WORLD);
  std::string thread_name = "RANK " + std::to_string(rank);
  loguru::set_thread_name(thread_name.c_str());
  {
    // --- Read mesh
    dolfinx::common::Timer t10("t10 Read the mesh");
    if (rank == 0)std::cout << "Start Reading Mesh" << std::endl;
    std::string filename=argv[1];
    const std::string mesh_file = filename+".xdmf";
    const auto [mesh_init, domain1_init, facet1_init] = read_mesh(mesh_file);
    if (rank == 0)std::cout << "Finished Reading Mesh ..." << std::endl;
    t10.stop();

    dolfinx::common::Timer t20("t20 Read the meshtie faces");
    if (rank == 0)std::cout << "Start Reading MeshTie faces" << std::endl;
    std::ifstream in(filename+".mtlst");
    std::vector<std::int32_t> mtie_faces;
    std::int32_t face1, face2;
    while(in >> face1 && in >> face2)
    {
        mtie_faces.push_back(face1);
        mtie_faces.push_back(face2);
    }
    in.close();    
    int nmtie = mtie_faces.size()/2; 
    if(rank==0)std::cout <<"Number of mesh tie pairs:"<<nmtie<<std::endl;
    t20.stop();

//    const std::int32_t contact_bdry_1 = 6; // top contact interface
//    const std::int32_t contact_bdry_2 = 13;  // bottom contact interface
    loguru::g_stderr_verbosity = loguru::Verbosity_OFF;
//    auto [mesh_new, facet1, domain1] = dolfinx_contact::create_contact_mesh(
//        *mesh_init, facet1_init, domain1_init,
//        {contact_bdry_1, contact_bdry_2}, 10.0);
    dolfinx::common::Timer t30("t30 Create contact mesh");
    if (rank == 0)std::cout << "Create Contact Mesh" << std::endl;
    auto [mesh_new, facet1, domain1] = dolfinx_contact::create_contact_mesh(
        *mesh_init, facet1_init, domain1_init, mtie_faces, 10.0);
    auto mesh = std::make_shared<dolfinx::mesh::Mesh<U>>(mesh_new);
    t30.stop();

    // Create function spaces
    dolfinx::common::Timer t40("t40 Create function spaces");
    if (rank == 0)std::cout << "Create Function Spaces" << std::endl;
    auto V = std::make_shared<fem::FunctionSpace<U>>(fem::create_functionspace(
        functionspace_form_linear_elasticity_J, "w", mesh));
    auto V0 = std::make_shared<fem::FunctionSpace<U>>(fem::create_functionspace(
        functionspace_form_linear_elasticity_J, "mu", mesh));
    t40.stop();

    double E = 115000;
    double E1 = 1000*E;
    double nu = 0.32;
    double gamma = 10;
    double theta = 0;

    double lmbda_val = E * nu / ((1 + nu) * (1 - nu));
    double mu_val = E / (2 * (1 + nu));

    // Create DG0 function for lame parameter lambda
    dolfinx::common::Timer t50("t50 Create DG0 function");
    if (rank == 0)std::cout << "Create DG0 Functions" << std::endl;

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
    t50.stop();

    // create integration domains for integrating over specific surfaces
    dolfinx::common::Timer t60("t60 Create integration domains");
    if (rank == 0)std::cout << "Create Integration Domains" << std::endl;
    auto facet_domains = fem::compute_integration_domains(
        fem::IntegralType::exterior_facet, *facet1.topology(), facet1.indices(),
        facet1.dim(), facet1.values());
    t60.stop();
    
    // Define variational forms
    dolfinx::common::Timer t70("t70 Create forms");
    if (rank == 0)std::cout << "Create Forms" << std::endl;
    auto J = std::make_shared<fem::Form<T>>(
        fem::create_form<T>(*form_linear_elasticity_J, {V, V},
                            {{"mu", mu}, {"lmbda", lmbda}}, {},
        {{dolfinx::fem::IntegralType::exterior_facet, facet_domains}}));
    auto F = std::make_shared<fem::Form<T>>(
        fem::create_form<T>(*form_linear_elasticity_F, {V},
                            {{"mu", mu}, {"lmbda", lmbda}}, {},
        {{dolfinx::fem::IntegralType::exterior_facet, facet_domains}}));
    t70.stop();

    // Define boundary conditions
    dolfinx::common::Timer t80("t80 Define dirichlet BCs");
    if (rank == 0)std::cout << "Define Dirichlet BCs" << std::endl;
    std::vector<std::shared_ptr<const dolfinx::fem::DirichletBC<T>>>bc{};
    const std::int32_t dirichlet_bdy1 = 25078; // first face
    auto facets1 = facet1.find(dirichlet_bdy1);
    auto bdofs1 = dolfinx::fem::locate_dofs_topological(
        *V->mesh()->topology_mutable(), *V->dofmap(), 2, facets1);
    auto bc1 = std::make_shared<const dolfinx::fem::DirichletBC<T>>(
        std::vector<T>({0.0, 0.0, 0.0}), bdofs1, V);
    bc.push_back(bc1);

    const std::int32_t dirichlet_bdy2 = 49517; // second face
    auto facets2 = facet1.find(dirichlet_bdy2);
    auto bdofs2 = dolfinx::fem::locate_dofs_topological(
        *V->mesh()->topology_mutable(), *V->dofmap(), 2, facets2);
    auto bc2 = std::make_shared<const dolfinx::fem::DirichletBC<T>>(
        std::vector<T>({0.0, 0.0, 0.0}), bdofs2, V);            
    bc.push_back(bc2); 
    t80.stop();
    
    // Create meshties
    dolfinx::common::Timer t90("t90 Create meshtie object");
    if (rank == 0)std::cout << "Create Meshtie Object" << std::endl;
    std::int32_t nmtie32 = nmtie; 
    std::vector<std::int32_t> offsets = {0, 2*nmtie32};
    auto contact_markers
        = std::make_shared<dolfinx::graph::AdjacencyList<std::int32_t>>(
            std::move(mtie_faces), std::move(offsets));

//    std::vector<std::int32_t> data = {contact_bdry_1, contact_bdry_2};
//    std::vector<std::int32_t> offsets = {0, 2};
//    auto contact_markers
//        = std::make_shared<dolfinx::graph::AdjacencyList<std::int32_t>>(
//            std::move(data), std::move(offsets));

    // wrap facet markers
    std::vector<std::shared_ptr<dolfinx::mesh::MeshTags<std::int32_t>>> markers
        = {std::make_shared<dolfinx::mesh::MeshTags<std::int32_t>>(facet1)};
    // define pairs (slave, master)
    std::vector<std::array<int, 2>> pairs;
    for(int imtie=0;imtie<nmtie;imtie++)
    {
       int p1 = 2*imtie;
       int p2 = 2*imtie+1;
       pairs.push_back({p1,p2});
       pairs.push_back({p2,p1});
    }  
//    std::vector<std::array<int, 2>> pairs = {{0, 1}, {1, 0}};
    auto meshties
        = dolfinx_contact::MeshTie(markers, contact_markers, pairs, mesh, 5);
    t90.stop();

    dolfinx::common::Timer t100("t100 Generate meshtie kernel data");
    if (rank == 0)std::cout << "Generate Meshtie Kernel Data" << std::endl;
    meshties.generate_kernel_data(dolfinx_contact::Problem::Elasticity, V,
                                  {{"mu", mu}, {"lambda", lmbda}}, E1 * gamma,
                                  theta);
    t100.stop();
    
    // Create matrix and vector
    dolfinx::common::Timer t110("t110 Create matrix & vector");
    if (rank == 0)std::cout << "Create Matrix & Vector" << std::endl;
    auto A = dolfinx::la::petsc::Matrix(
        meshties.create_petsc_matrix(*J, std::string()), false);
    dolfinx::la::Vector<T> b(F->function_spaces()[0]->dofmap()->index_map,
                             F->function_spaces()[0]->dofmap()->index_map_bs());
    t110.stop();

    // Assemble vector
    dolfinx::common::Timer t120("t120 Assemble vector");
    if (rank == 0)std::cout << "Assemble Vector" << std::endl;
    b.set(0.0);
    meshties.assemble_vector(b.mutable_array(), V,
                             dolfinx_contact::Problem::Elasticity);
    dolfinx::fem::assemble_vector(b.mutable_array(), *F);
    dolfinx::fem::apply_lifting<T, U>(b.mutable_array(), {J}, {{bc}}, {},
                                      double(1.0));
    b.scatter_rev(std::plus<T>());
    dolfinx::fem::set_bc<T, U>(b.mutable_array(), {bc});
    t120.stop();
    
    // Assemble matrix
    dolfinx::common::Timer t130("t130 Assemble matrix");
    if (rank == 0)std::cout << "Assemble Matrix" << std::endl;
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
    t130.stop();
    
    // Build near-nullspace and attach to matrix
    dolfinx::common::Timer t140("t140 create null space");
    if (rank == 0)std::cout << "Create Null Space" << std::endl;
    MatNullSpace ns = build_near_nullspace(*V);
    MatSetNearNullSpace(A.mat(), ns);
    MatNullSpaceDestroy(&ns);
    t140.stop();

    // Set up linear solver with parameters
    dolfinx::common::Timer t150("t150 Solver setup");
    if (rank == 0)std::cout << "Sover Setup" << std::endl;
    dolfinx::common::Timer mshty8("mshty8 linear solve");
    dolfinx::la::petsc::KrylovSolver ksp(MPI_COMM_WORLD);
    ksp.set_options_prefix("st_");
    ksp.set_from_options();
    ksp.set_operator(A.mat());
    t150.stop();
    
    // displacement function
    dolfinx::common::Timer t160("t160 create solution vector");
    if (rank == 0)std::cout << "Create Solution Vector" << std::endl;
    auto u = std::make_shared<fem::Function<T>>(V);
    dolfinx::la::petsc::Vector _u(
        dolfinx::la::petsc::create_vector_wrap(*u->x()), false);
    dolfinx::la::petsc::Vector _b(dolfinx::la::petsc::create_vector_wrap(b),
                                  false);
    t160.stop(); 

    dolfinx::common::Timer t170("t170 solve linear system");
    if (rank == 0)std::cout << "Solve Linear System" << std::endl;
    // solve linear system
    ksp.solve(_u.vec(), _b.vec());
    t170.stop();

    // Update ghost values before output
    dolfinx::common::Timer t180("t180 write output data");
    if (rank == 0)std::cout << "Write Output Data" << std::endl;
    u->x()->scatter_fwd();
    loguru::g_stderr_verbosity = loguru::Verbosity_OFF;
    dolfinx::io::XDMFFile file2(mesh->comm(), "result.xdmf", "w");
    file2.write_mesh(*mesh);
    file2.write_function(*u, 0.0);
    file2.close();
    t180.stop();
    list_timings(V->mesh()->comm(), {TimingType::wall});
  }

  PetscFinalize();
  return 0;
}
