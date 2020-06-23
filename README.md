# ExaCLAMR

Re-Implementation of the Shallow Water Solver LANL/CLAMR using Kokkos, Cabana, and Cajita.

## Current Status

- Properly functions using
  - Serial
  - OpenMP
  - CudaUVM
  - Serial + MPI
  - OpenMP + MPI
  - CudaUVM + MPI
- Able to write to Silo files using PMPIO (Tested on Mac in Serial, OpenMP, Serial + MPI, OpenMP + MPI)
- Can Visualize results using VisIt

## Future Directions and Tasks

- [ ] Fix Cuda MPI issue in use of Cajita Halo gather (currently functioning with a work around using CudaUVM and a custom halo exchange). It might be an issue with how ExaCLAMR is using the Cajita gather?
- [ ] Get Silo to build on Wheeler and Xena and get it working with the CudaUVM case
- [ ] Template the mesh and problem manager classes for regular grids, AMR grids, and space-filling curves
- [ ] Investigate using Cabana AoSoA
- [ ] Add particle physics

### Building on UNM CARC Wheeler

```bash
git clone https://github.com/CUP-ECS/ExaCLAMR.git
cd ExaCLAMR
bash scripts/build_wheeler.sh
```

### Building on UNM CARC Xena

```bash
git clone https://github.com/CUP-ECS/ExaCLAMR.git
cd ExaCLAMR
bash scripts/build_xena.sh
```

### Performance