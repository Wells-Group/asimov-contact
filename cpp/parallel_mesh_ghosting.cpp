// Copyright (C) 2023 Chris N. Richardson
//
// This file is part of DOLFINx_CONTACT
//
// SPDX-License-Identifier:    MIT

#include <dolfinx/common/MPI.h>
#include <dolfinx/common/Timer.h>
#include <dolfinx/common/log.h>
#include <dolfinx/graph/AdjacencyList.h>
#include <dolfinx/mesh/utils.h>
#include <mpi.h>

#include "parallel_mesh_ghosting.h"
#include "point_cloud.h"

#include <set>

namespace
{
std::pair<std::vector<std::int64_t>, std::vector<int>>
copy_to_all(MPI_Comm comm, const std::vector<std::int64_t>& indices,
            std::span<const int> values, int nv)
{
  int idx_size = indices.size();
  int mpi_size = dolfinx::MPI::size(comm);
  std::vector<int> recv_count(mpi_size), recv_offset(mpi_size + 1);
  MPI_Allgather(&idx_size, 1, MPI_INT, recv_count.data(), 1, MPI_INT, comm);
  for (int i = 0; i < mpi_size; ++i)
    recv_offset[i + 1] = recv_offset[i] + recv_count[i];

  std::vector<std::int64_t> all_indices(recv_offset.back());
  MPI_Allgatherv(indices.data(), indices.size(), MPI_INT64_T,
                 all_indices.data(), recv_count.data(), recv_offset.data(),
                 MPI_INT64_T, comm);

  // Change count for data (one item per facet)
  std::for_each(recv_count.begin(), recv_count.end(),
                [nv](int& n) { n /= nv; });
  std::for_each(recv_offset.begin(), recv_offset.end(),
                [nv](int& n) { n /= nv; });

  std::vector<int> all_values(recv_offset.back());
  MPI_Allgatherv(values.data(), values.size(), MPI_INT, all_values.data(),
                 recv_count.data(), recv_offset.data(), MPI_INT, comm);

  return {std::move(all_indices), std::move(all_values)};
};

} // namespace

std::tuple<dolfinx::mesh::Mesh<double>, dolfinx::mesh::MeshTags<std::int32_t>,
           dolfinx::mesh::MeshTags<std::int32_t>>
create_contact_mesh(dolfinx::mesh::Mesh<double>& mesh,
                    const dolfinx::mesh::MeshTags<std::int32_t>& fmarker,
                    const dolfinx::mesh::MeshTags<std::int32_t>& cmarker,
                    const std::vector<std::int32_t>& tags, double R = 0.2)
{
  LOG(WARNING) << "Create Contact Mesh";

  // FIX: function too long - break up

  int tdim = mesh.topology()->dim();
  int num_cell_vertices
      = dolfinx::mesh::num_cell_vertices(mesh.topology()->cell_types()[0]);
  dolfinx::mesh::CellType facet_type
      = dolfinx::mesh::cell_facet_type(mesh.topology()->cell_types()[0], 0);
  int num_facet_vertices = dolfinx::mesh::num_cell_vertices(facet_type);

  // Get cells attached to marked facets
  mesh.topology()->create_connectivity(tdim - 1, tdim);
  mesh.topology()->create_connectivity(tdim, 0);
  auto fc = mesh.topology()->connectivity(tdim - 1, tdim);
  auto fv = mesh.topology()->connectivity(tdim - 1, 0);
  auto cv = mesh.topology()->connectivity(tdim, 0);

  // Extract facet markers with given tags
  std::vector<int> marker_subset;
  {
    const auto& mv = fmarker.values();
    const auto& mi = fmarker.indices();
    for (std::size_t i = 0; i < mi.size(); ++i)
    {
      for (std::int32_t k : tags)
      {
        if (k == mv[i])
        {
          marker_subset.push_back(mi[i]);
          break;
        }
      }
    }
  }

  LOG(WARNING) << "Compute cell destinations";

  // Find destinations for the cells attached to the tag-marked facets
  auto cell_dests = dolfinx_contact::compute_ghost_cell_destinations(
      mesh, marker_subset, R);

  LOG(WARNING) << "cells to ghost";

  std::vector<int> cells_to_ghost;
  for (std::int32_t f : marker_subset)
    cells_to_ghost.push_back(fc->links(f)[0]);

  std::map<int, std::vector<int>> cell_to_dests;
  for (std::size_t i = 0; i < cells_to_ghost.size(); ++i)
  {
    int c = cells_to_ghost[i];
    cell_to_dests[c] = std::vector<int>(cell_dests.links(i).begin(),
                                        cell_dests.links(i).end());
  }

  int ncells = mesh.topology()->index_map(tdim)->size_local();

  // Convert marked facets to list of (global) vertices for each facet
  std::vector<int> local_indices;
  for (auto f : fmarker.indices())
  {
    auto fl = fv->links(f);
    local_indices.insert(local_indices.end(), fl.begin(), fl.end());
  }
  std::vector<std::int64_t> fv_indices(local_indices.size());
  mesh.topology()->index_map(0)->local_to_global(local_indices, fv_indices);
  for (std::size_t i = 0; i < fv_indices.size(); i += num_facet_vertices)
  {
    std::sort(std::next(fv_indices.begin(), i),
              std::next(fv_indices.begin(), i + num_facet_vertices));
  }
  local_indices.clear();

  // Convert marked cells to list of (global) vertices for each cell
  for (auto c : cmarker.indices())
  {
    auto cl = cv->links(c);
    local_indices.insert(local_indices.end(), cl.begin(), cl.end());
  }
  std::vector<std::int64_t> cv_indices(local_indices.size());
  mesh.topology()->index_map(0)->local_to_global(local_indices, cv_indices);
  for (std::size_t i = 0; i < cv_indices.size(); i += num_cell_vertices)
  {
    std::sort(std::next(cv_indices.begin(), i),
              std::next(cv_indices.begin(), i + num_cell_vertices));
  }

  LOG(WARNING) << "Copy markers to other processes";

  // Copy facets and markers to all processes
  auto [all_facet_indices, all_facet_values] = copy_to_all(
      mesh.comm(), fv_indices, fmarker.values(), num_facet_vertices);
  // Repeat for cell data
  auto [all_cell_indices, all_cell_values] = copy_to_all(
      mesh.comm(), cv_indices, cmarker.values(), num_cell_vertices);

  // Convert topology to global indexing, and restrict to non-ghost cells
  std::vector<int> topo = mesh.topology()->connectivity(tdim, 0)->array();
  // Cut off any ghost vertices
  topo.resize(ncells * num_cell_vertices);
  std::vector<std::int64_t> topo_global(topo.size());
  mesh.topology()->index_map(0)->local_to_global(topo, topo_global);
  dolfinx::graph::AdjacencyList<std::int64_t> topo_adj
      = dolfinx::graph::regular_adjacency_list(topo_global, num_cell_vertices);

  std::size_t num_vertices = mesh.topology()->index_map(0)->size_local();
  std::size_t gdim = mesh.geometry().dim();

  std::array<std::size_t, 2> xshape = {num_vertices, gdim};
  std::vector<double> x;
  x.reserve(num_vertices * gdim);
  std::span<const double> xg = mesh.geometry().x();
  for (int i = 0; i < num_vertices; ++i)
  {
    for (int j = 0; j < gdim; ++j)
      x.push_back(xg[i * 3 + j]);
  }

  auto partitioner
      = [cell_to_dests,
         ncells](MPI_Comm comm, int nparts, int tdim,
                 const dolfinx::graph::AdjacencyList<std::int64_t>& cells)
  {
    int rank = dolfinx::MPI::rank(comm);
    std::vector<std::int32_t> dests;
    std::vector<int> offsets = {0};
    for (int c = 0; c < ncells; ++c)
    {
      dests.push_back(rank);
      if (auto it = cell_to_dests.find(c); it != cell_to_dests.end())
        dests.insert(dests.end(), it->second.begin(), it->second.end());

      // Ghost to other processes
      offsets.push_back(dests.size());
    }
    return dolfinx::graph::AdjacencyList<std::int32_t>(std::move(dests),
                                                       std::move(offsets));
  };

  LOG(WARNING) << "Repartition";
  dolfinx::common::Timer trepart("~Contact: Add ghosts: Repartition");
  auto new_mesh = dolfinx::mesh::create_mesh(
      mesh.comm(), topo_adj, mesh.geometry().cmaps(), x, xshape, partitioner);
  trepart.stop();

  LOG(WARNING) << "Remap markers on new mesh";

  dolfinx::common::Timer tremap(
      "~Contact: Add ghosts: Remap markers on new mesh");
  // Remap vertices back to input indexing
  // This is rather messy, we need to map vertices to geometric nodes
  // then back to original index
  auto global_remap = new_mesh.geometry().input_global_indices();
  int nv = new_mesh.topology()->index_map(0)->size_local()
           + new_mesh.topology()->index_map(0)->num_ghosts();
  std::vector<std::int32_t> nvrange(nv);
  std::iota(nvrange.begin(), nvrange.end(), 0);
  auto vert_to_geom = entities_to_geometry(new_mesh, 0, nvrange, false);

  // Recreate facets
  new_mesh.topology()->create_entities(tdim - 1);
  new_mesh.topology()->create_connectivity(tdim - 1, tdim);
  new_mesh.topology()->create_connectivity(tdim, 0);

  // Create a list of all facet - vertices(original global index)
  auto fv_new = new_mesh.topology()->connectivity(tdim - 1, 0);
  int num_new_fv = fv_new->num_nodes();
  std::vector<std::int64_t> fv_new_indices(fv_new->array().begin(),
                                           fv_new->array().end());

  // Map back to original index
  std::for_each(fv_new_indices.begin(), fv_new_indices.end(),
                [&](std::int64_t& idx)
                { idx = global_remap[vert_to_geom[idx]]; });

  // Sort each facet into order for comparison
  for (int i = 0; i < num_new_fv; ++i)
  {
    std::sort(std::next(fv_new_indices.begin(), i * num_facet_vertices),
              std::next(fv_new_indices.begin(), (i + 1) * num_facet_vertices));
  }

  LOG(WARNING) << "Lex match facet markers";
  dolfinx::common::Timer tlex1("~Contact: Add ghosts: Lex match facet markers");

  auto [new_fm_index, new_fm_data] = dolfinx_contact::lex_match(
      num_new_fv, fv_new_indices, all_facet_indices, all_facet_values);

  auto new_fmarker = dolfinx::mesh::MeshTags<std::int32_t>(
      new_mesh.topology(), tdim - 1, new_fm_index, new_fm_data);

  tlex1.stop();

  // Create a list of all cell - vertices(original global index)
  auto cv_new = new_mesh.topology()->connectivity(tdim, 0);

  int num_new_cv = cv_new->num_nodes();
  std::vector<std::int64_t> cv_new_indices(cv_new->array().begin(),
                                           cv_new->array().end());

  // Map back to original index
  std::for_each(cv_new_indices.begin(), cv_new_indices.end(),
                [&](std::int64_t& idx)
                { idx = global_remap[vert_to_geom[idx]]; });

  // Sort each cell into order for comparison
  for (int i = 0; i < num_new_cv; ++i)
  {
    std::sort(std::next(cv_new_indices.begin(), i * num_cell_vertices),
              std::next(cv_new_indices.begin(), (i + 1) * num_cell_vertices));
  }

  LOG(WARNING) << "Lex match cell markers";
  dolfinx::common::Timer tlex2("~Contact: Add ghosts: Lex match cell markers");

  auto [new_cm_index, new_cm_data] = dolfinx_contact::lex_match(
      num_new_cv, cv_new_indices, all_cell_indices, all_cell_values);

  auto new_cmarker = dolfinx::mesh::MeshTags<std::int32_t>(
      new_mesh.topology(), tdim, new_cm_index, new_cm_data);

  tlex2.stop();

  return {new_mesh, new_fmarker, new_cmarker};
}

dolfinx::graph::AdjacencyList<std::int32_t>
dolfinx_contact::compute_ghost_cell_destinations(
    const dolfinx::mesh::Mesh<double>& mesh,
    std::span<const std::int32_t> marker_subset, double R)
{
  // For each marked facet, given by indices in "marker_subset", get the
  // list of processes which the attached cell should be sent to, for
  // ghosting. Neighbouring facets within distance "R".
  LOG(WARNING) << "Compute ghost cell destinations";

  const int size = dolfinx::MPI::size(mesh.comm());
  const int rank = dolfinx::MPI::rank(mesh.comm());

  // 1. Get midpoints of all facets on interfaces
  const int tdim = mesh.topology()->dim();

  auto x = mesh.geometry().x();
  std::vector<std::int32_t> facet_to_geom
      = entities_to_geometry(mesh, tdim - 1, marker_subset, false);
  const int num_facets = marker_subset.size();
  std::vector<double> facet_midpoint;
  facet_midpoint.reserve(num_facets * 3);
  if (num_facets > 0)
  {
    const int nv_per_facet = facet_to_geom.size() / num_facets;
    std::array<double, 3> midpoint;
    for (int i = 0; i < num_facets; ++i)
    {
      midpoint = {0, 0, 0};
      for (int j = 0; j < nv_per_facet; ++j)
      {
        int vidx = facet_to_geom[i * nv_per_facet + j] * 3;
        for (int k = 0; k < 3; ++k)
          midpoint[k] += x[vidx + k] / nv_per_facet;
      }
      facet_midpoint.insert(facet_midpoint.end(), midpoint.begin(),
                            midpoint.end());
    }
  }

  // 2. Send midpoints to process zero
  int count = facet_midpoint.size();
  std::vector<int> all_counts;
  if (rank == 0)
    all_counts.resize(size);
  MPI_Gather(&count, 1, MPI_INT, all_counts.data(), 1, MPI_INT, 0, mesh.comm());
  std::vector<int> offsets = {0};
  for (auto c : all_counts)
    offsets.push_back(offsets.back() + c);

  std::vector<double> x_all_flat(offsets.back());
  MPI_Gatherv(facet_midpoint.data(), facet_midpoint.size(), MPI_DOUBLE,
              x_all_flat.data(), all_counts.data(), offsets.data(), MPI_DOUBLE,
              0, mesh.comm());

  // For each facet, get a list of neighbor processes
  // These are only built on process zero
  std::vector<int> nbr_procs;
  std::vector<int> nbr_offsets = {0};

  if (rank == 0)
  {
    LOG(WARNING) << "Point cloud search on root process";

    std::for_each(offsets.begin(), offsets.end(), [](int& i) { i /= 3; });

    // Find all pairs of facets within radius R
    auto x_near = dolfinx_contact::point_cloud_pairs(x_all_flat, R);

    int i = 0;
    std::vector<int> neighbor_p;
    std::vector<int> pr;

    for (int p = 0; p < size; ++p)
    {
      assert(all_counts[p] % 3 == 0);
      const int num_facets_p = all_counts[p] / 3;

      // Reserve space for 'offsets' for this process
      neighbor_p.resize(num_facets_p + 1, 0);

      for (int j = 0; j < num_facets_p; ++j)
      {
        pr.clear();
        for (int n : x_near.links(i))
        {
          // Find which process this facet came from
          int q = std::distance(
                      offsets.begin(),
                      std::upper_bound(offsets.begin(), offsets.end(), n))
                  - 1;

          // Add to the sendback list, if not the same process
          if (q != p)
          {
            if (std::find(pr.begin(), pr.end(), q) == pr.end())
            {
              pr.push_back(q);
              std::sort(pr.begin(), pr.end());
            }
          }
        }
        neighbor_p.insert(neighbor_p.end(), pr.begin(), pr.end());
        neighbor_p[j + 1] = neighbor_p.size() - (num_facets_p + 1);
        ++i;
      }
      nbr_procs.insert(nbr_procs.end(), neighbor_p.begin(), neighbor_p.end());
      nbr_offsets.push_back(nbr_procs.size());
    }
  }

  // Scatter back sharing data to original process
  std::vector<int> dsizes(size);
  for (int i = 0; i < size; ++i)
    dsizes[i] = nbr_offsets[i + 1] - nbr_offsets[i];
  int my_recv_size;
  MPI_Scatter(dsizes.data(), 1, MPI_INT, &my_recv_size, 1, MPI_INT, 0,
              mesh.comm());

  std::vector<int> my_recv_data(my_recv_size);
  MPI_Scatterv(nbr_procs.data(), dsizes.data(), nbr_offsets.data(), MPI_INT,
               my_recv_data.data(), my_recv_size, MPI_INT, 0, mesh.comm());

  // Unpack received data to get additional destinations for each facet /
  // cell
  std::vector<int> doffsets(my_recv_data.begin(),
                            std::next(my_recv_data.begin(), num_facets + 1));

  std::vector<int> cell_dests(std::next(my_recv_data.begin(), num_facets + 1),
                              my_recv_data.end());

  return dolfinx::graph::AdjacencyList<std::int32_t>(cell_dests, doffsets);
}

std::pair<std::vector<int>, std::vector<int>>
dolfinx_contact::lex_match(int dim,
                           const std::vector<std::int64_t>& local_indices,
                           const std::vector<std::int64_t>& in_indices,
                           const std::vector<std::int32_t>& in_values)
{
  LOG(WARNING) << "Lex match: [" << local_indices.size() << ", "
               << in_indices.size() << ", " << in_values.size() << "]";

  assert(local_indices.size() % dim == 0);
  assert(in_indices.size() % dim == 0);
  assert(in_values.size() == in_indices.size() / dim);

  // Get the permutation that sorts local_indices into order
  LOG(WARNING) << "Sort p";
  std::vector<int> p_local(local_indices.size() / dim);
  std::iota(p_local.begin(), p_local.end(), 0);
  std::sort(p_local.begin(), p_local.end(),
            [&](int a, int b)
            {
              return std::lexicographical_compare(
                  local_indices.begin() + dim * a,
                  local_indices.begin() + dim * (a + 1),
                  local_indices.begin() + dim * b,
                  local_indices.begin() + dim * (b + 1));
            });

  // Get the permutation that sorts in_indices into order
  LOG(WARNING) << "Sort q";
  std::vector<int> p_in(in_indices.size() / dim);
  std::iota(p_in.begin(), p_in.end(), 0);
  std::sort(
      p_in.begin(), p_in.end(),
      [&](int a, int b)
      {
        return std::lexicographical_compare(
            in_indices.begin() + dim * a, in_indices.begin() + dim * (a + 1),
            in_indices.begin() + dim * b, in_indices.begin() + dim * (b + 1));
      });

  LOG(WARNING) << p_in.size() << "," << p_local.size();

  std::vector<std::pair<int, int>> new_markers;
  std::size_t i = 0;
  std::size_t j = 0;

  // Go through both sets of indices in the same order
  while (i < p_in.size() and j < p_local.size())
  {
    int a = p_in[i] * dim;
    int b = p_local[j] * dim;
    // Matching: found the same entity in both lists; save marker
    if (std::equal(in_indices.begin() + a, in_indices.begin() + a + dim,
                   local_indices.begin() + b))
    {
      new_markers.push_back({p_local[j], in_values[p_in[i]]});
      ++i;
      ++j;
    }
    else
    {
      // Not matching: increment pointer in the list with the "lower" entity
      bool fwdi = std::lexicographical_compare(
          in_indices.begin() + a, in_indices.begin() + a + dim,
          local_indices.begin() + b, local_indices.begin() + b + dim);

      if (fwdi)
        ++i;
      else
        ++j;
    }
  }

  // Clean up new markers
  std::sort(new_markers.begin(), new_markers.end());
  auto last = std::unique(new_markers.begin(), new_markers.end());
  new_markers.erase(last, new_markers.end());

  std::vector<std::int32_t> nmi, nmv;
  nmi.reserve(new_markers.size());
  nmv.reserve(new_markers.size());
  for (auto p : new_markers)
  {
    nmi.push_back(p.first);
    nmv.push_back(p.second);
  }

  return {std::move(nmi), std::move(nmv)};
}
