// Copyright (C) 2021-2024 Jørgen S. Dokken, Sarah Roggendorf
//
// This file is part of DOLFINx_CONTACT
//
// SPDX-License-Identifier:    MIT

#pragma once

#include <dolfinx_contact/utils.h>
#include <functional>

namespace contact_wrappers
{
/// This class wraps kernels from C++ for use in nanobind, as nanobind
/// automatically wraps std::function of pointers to ints, which in turn
/// cannot be transferred back to C++
template <typename T>
class KernelWrapper
{
public:
  /// Wrap a Kernel
  KernelWrapper(dolfinx_contact::kernel_fn<T> kernel) : _kernel(kernel) {}

  /// Assignment operator
  KernelWrapper& operator=(dolfinx_contact::kernel_fn<T> kernel)
  {
    this->_kernel = kernel;
    return *this;
  }

  /// Get the C++ kernel
  dolfinx_contact::kernel_fn<T> get() { return _kernel; }

private:
  // The underlying communicator
  dolfinx_contact::kernel_fn<T> _kernel;
};

} // namespace contact_wrappers
