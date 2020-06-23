# ExaCLAMR

Re-Implementation of the Shallow Water Solver LANL/CLAMR using Kokkos, Cabana, and Cajita.

## Current Status

- Can function properly using
  - Serial
  - OpenMP
  - CudaUVM
  - Serial + MPI
  - OpenMP + MPI
  - CudaUVM + MPI
- Able to write to Silo files using PMPIO 
- Can Visualize results using VisIt
- Tested on Regular Grid Sizes up to 14,000 x 14,000

## Future Directions and Tasks

- [ ] Fix Cuda MPI issue in use of Cajita Halo gather (currently functioning with a work around using CudaUVM and a custom halo exchange). It might be an issue with how ExaCLAMR is using the Cajita gather?
- [ ] Template the mesh and problem manager classes for regular grids, AMR grids, and space-filling curves
- [ ] Investigate using Cabana AoSoA
- [ ] Add a particle solver

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