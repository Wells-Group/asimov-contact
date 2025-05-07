# dolfinx-contact

This library is for computing contact between deforming bodies using
[FEniCSx](https://fenicsproject.org), and targets massively parallel
simulations. It builds on the
[DOLFINx](https://github.com/FEniCS/dolfinx) library.

dolfinx-contact is under heavy development and is highly experimental.

## Installation
DOLFINx contact requires DOLFINx (v0.9.0) installed on your system.
To build the library, you can call:
```bash

```bash
export VERBOSE=1
cmake -G Ninja -DCMAKE_BUILD_TYPE=Developer  -B build-contact -S cpp/
ninja -C build-contact install
```
to install the C++ interface
and
```bash
python3 -m pip -v install -r ./python/build-requirements.txt
python3 -m pip -v install --no-build-isolation python/
```
to install the Python interface.

## Notes

Contact models using DOLFINx and Nitsche's method.

See demos for comparisons of traditional Nitche methods compared to SNES
models for unilateral contact.

See tests for testing of custom assemblers that will be used for
unbiased contact between two mesh surfaces.
