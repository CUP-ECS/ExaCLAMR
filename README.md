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
- [x] Template the mesh and problem manager classes for regular grids, AMR grids
- [ ] Investigate using Cabana AoSoA
- [ ] Add particle physics

## Building ExaCLAMR
ExaCLAKMR is built and installed using cmake, and relies on variety of packages and
libraries:
  * Kokkos - Programming system for accelerated and threaded architectures
  * Cabana/Cajita - Framework for regular mesh and particle oprogramming in Kokkos
  * Silo - Parallel I/O library for reading in restarts and writing output for
    visualization using VisIt or Paraview
  * HeFFTE - GPU-accelerated high-speed fast fourier transform library (Note: not
    needed by ExaCLAMR but we'll want it for the z-model implementation)

### Building with a Spack environment

Generally, the easiest way to build ExaCLAMR is to use [Spack](http://spack.io) to
either install and load the prerequisites or to create a dedicated environment
for building and running AppName. The spack specification (etc/spack.yaml) is
included to create a spack environment in which the build and run AppName. Note, however
//that there are important caveats at the top of this file you'll ened to pay attention to
to use it//. In particular, to use it to build a CUDA environment you'll need to patch your
cabana/packages.py spec and set teh cuda architecture kokkos uses. Alternatively, you
can just remove the cuda specifiers from the spack.yaml file.

To create a spack environment for compiling and running ExaCLAMR in a build directory:

```
prompt> mkdir build; cd build
prompt> spack env create -d . /path/to/ExaCLAMR/etc/spack.yaml
prompt> spack env activate .
prompt> spack concretize
prompt> spack install
```

Once all ExaCLAMR dependencies are installed either manually or via a
Spack environment, use cmake to build it from the chosen build
directory:
```
cmake /path/to/ExaCLAMR
```

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

- Note: You have to build on a compute node with a GPU

```bash
git clone https://github.com/CUP-ECS/ExaCLAMR.git
cd ExaCLAMR
bash scripts/build_xena.sh -a
mkdir -p data
mkdir -p data/raw
```

### Running on UNM CARC Xena

```bash
sbatch scripts/xena_run.sh
```

## Performance
