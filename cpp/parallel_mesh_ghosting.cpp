// Copyright (C) 2023 Chris N. Richardson
//
// This file is part of DOLFINx_CONTACT
//
// SPDX-License-Identifier:    MIT

#include <dolfinx/common/MPI.h>
#include <dolfinx/common/log.h>
#include <dolfinx/graph/AdjacencyList.h>
#include <dolfinx/mesh/utils.h>

#include "parallel_mesh_ghosting.h"
#include "point_cloud.h"

#include <set>

dolfinx::graph::AdjacencyList<std::int32_t>
dolfinx_contact::compute_ghost_cell_destinations(
    const dolfinx::mesh::Mesh<double>& mesh,
    std::span<const std::int32_t> marker_subset, double R)
{
  // For each marked facet, given by indices in "marker_subset", get the list of
  // processes which the attached cell should be sent to, for ghosting.
  // Neighbouring facets within distance "R".
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

  // Unpack received data to get additional destinations for each facet / cell
  std::vector<int> doffsets(my_recv_data.begin(),
                            std::next(my_recv_data.begin(), num_facets + 1));

  std::vector<int> cell_dests(std::next(my_recv_data.begin(), num_facets + 1),
                              my_recv_data.end());

  return dolfinx::graph::AdjacencyList<std::int32_t>(cell_dests, doffsets);
}

std::vector<std::pair<int, int>>
dolfinx_contact::lex_match(int dim, std::vector<std::int32_t>& local_indices,
                           std::vector<std::int32_t>& in_indices,
                           std::vector<std::int32_t>& in_values)
{
  LOG(WARNING) << "Lex match";

  assert(local_indices.size() % dim == 0);
  assert(in_indices.size() % dim == 0);
  assert(in_values.size() == in_indices.size() / dim);

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
  int i = 0;
  int j = 0;

  while (i < p_in.size() and j < p_local.size())
  {
    int a = p_in[i] * dim;
    int b = p_local[j] * dim;
    if (std::equal(in_indices.begin() + a, in_indices.begin() + a + dim,
                   local_indices.begin() + b))
    {
      new_markers.push_back({p_local[j], in_values[p_in[i]]});
      ++i;
      ++j;
    }
    else
    {
      bool fwdi = std::lexicographical_compare(
          in_indices.begin() + a, in_indices.begin() + a + dim,
          local_indices.begin() + b, local_indices.begin() + b + dim);

      if (fwdi)
        ++i;
      else
        ++j;
    }
  }

  std::sort(new_markers.begin(), new_markers.end());
  auto last = std::unique(new_markers.begin(), new_markers.end());
  new_markers.erase(last, new_markers.end());

  return new_markers;
}
