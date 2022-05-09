// Copyright (C) 2021 Jørgen S. Dokken, Sarah Roggendorf
//
// This file is part of DOLFINx_CONTACT
//
// SPDX-License-Identifier:    MIT
#include <caster_petsc.h>
#include <dolfinx_contact/utils.h>
#include <functional>

namespace contact_wrappers
{

/// This class wraps kernels from C++ for use in pybind11,
/// as pybind automatically wraps std::function of pointers to ints,
/// which in turn cannot be transferred back to C++

class KernelWrapper
{
public:
  /// Wrap a Kernel
  KernelWrapper(dolfinx_contact::kernel_fn<PetscScalar> kernel)
      : _kernel(kernel)
  {
  }

  /// Assignment operator
  KernelWrapper& operator=(dolfinx_contact::kernel_fn<PetscScalar> kernel)
  {
    this->_kernel = kernel;
    return *this;
  }

  /// Get the C++ kernel
  dolfinx_contact::kernel_fn<PetscScalar> get() { return _kernel; }

private:
  // The underlying communicator
  dolfinx_contact::kernel_fn<PetscScalar> _kernel;
};

} // namespace contact_wrappers
