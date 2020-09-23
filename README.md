# ExaCLAMR

Re-Implementation of the Shallow Water Solver LANL/CLAMR using Kokkos, Cabana, and Cajita.

## Current Status

- Properly functions using
  - Serial
  - OpenMP
  - Cuda
  - Serial + MPI
  - OpenMP + MPI
  - Cuda + MPI
- Able to write to Silo files using PMPIO (Tested on Mac in Serial, OpenMP, Serial + MPI, OpenMP + MPI)
- Can Visualize results using VisIt

## Future Directions and Tasks

- [x] Fix Cuda MPI issue in use of Cajita Halo gather (currently functioning with a work around using CudaUVM and a custom halo exchange). It might be an issue with how ExaCLAMR is using the Cajita gather?
- [x] Get Silo to build on Wheeler and Xena and get it working with the CudaUVM case
- [x] Template the mesh and problem manager classes for regular grids, AMR grids, and space-filling curves (see Jered branch)
- [x] 2-D Hilbert Layout
- [ ] Investigate using Cabana AoSoA
- [ ] Add particle physics

### Building on UNM CARC Wheeler

```bash
git clone https://github.com/CUP-ECS/ExaCLAMR.git
cd ExaCLAMR
bash scripts/build_wheeler.sh -a
mkdir -p data
mkdir -p data/raw
```

### Running on UNM CARC Wheeler - Example on 2 Ranks (Serial, MPI+OpenMP)
```bash
module load cmake-3.15.4-gcc-8.3.0-rmxifnl
module load openmpi-3.1.4-gcc-8.3.0-w3pkrvv
module load gcc-8.3.0-gcc-4.8.5-wwpinbr
module load hypre-2.14.0-gcc-7.3.0-openmpi-mkl-zndhsgh
mpirun -np 2 --display-map --map-by ppr:1:node --bind-to none -machinefile $PBS_NODEFILE -x PATH -x LD_LIBRARY_PATH ./build/examples/DamBreak
mpirun -np 2 --display-map --map-by ppr:1:node --bind-to none -machinefile $PBS_NODEFILE -x PATH -x LD_LIBRARY_PATH ./build/examples/DamBreak -mopenmp
```

### Building on UNM CARC Xena

- Note: You have to build on a compute node

```bash
git clone https://github.com/CUP-ECS/ExaCLAMR.git
cd ExaCLAMR
bash scripts/build_xena.sh -a
mkdir -p data
mkdir -p data/raw
```

### Performance
