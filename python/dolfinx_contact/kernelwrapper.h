// Copyright (C) 2021 Jørgen S. Dokken, Sarah Roggendorf
//
// This file is part of DOLFINx_CONTACT
//
// SPDX-License-Identifier:    MIT
#include <caster_petsc.h>
#include <functional>

using contact_kernel_fn = std::function<void(
    std::vector<std::vector<PetscScalar>>&, const double*, const double*,
    const double*, const int*, const std::uint8_t*, const std::int32_t)>;

namespace contact_wrappers
{

/// This class wraps kernels from C++ for use in pybind11,
/// as pybind automatically wraps std::function of pointers to ints,
/// which in turn cannot be transferred back to C++

class KernelWrapper
{
public:
  /// Wrap a Kernel
  KernelWrapper(contact_kernel_fn kernel) : _kernel(kernel) {}

  /// Assignment operator
  KernelWrapper& operator=(contact_kernel_fn kernel)
  {
    this->_kernel = kernel;
    return *this;
  }

  /// Get the C++ kernel
  contact_kernel_fn get() { return _kernel; }

private:
  // The underlying communicator
  contact_kernel_fn _kernel;
};
} // namespace contact_wrappers
