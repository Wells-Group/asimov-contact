# dolfinx-contact

This library is for computing contact between deforming bodies using
[FEniCSx](https://fenicsproject.org), and targets massively parallel
simulations. It builds on the
[DOLFINx](https://github.com/FEniCS/dolfinx) library.

dolfinx-contact is under heavy development and is highly experimental.


## Notes

Contact models using DOLFINx and Nitsche's method.

See demos for comparisons of traditional Nitche methods compared to SNES
models for unilateral contact.

See tests for testing of custom assemblers that will be used for
unbiased contact between two mesh surfaces.
