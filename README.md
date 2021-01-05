# p2z-tests

## OpenMP
#### Compilers: gcc, icc, pgi.
version 3 is most up to date. 
$ make COMPILER=gcc MODE=omp
$ make COMPILER=icc MODE=omp
$ make COMPILER=pgi MODE=omp


## OpenACC
#### Compile with PGI
//Compile OpenACC C++ version
$ make COMPILER=pgi MODE=acc SRCTYPE=cpp
//Compile OpenACC C version
$ make COMPILER=pgi MODE=acc SRCTYPE=c

## Tbb
#### Compilers: gcc, icc
$ make COMPILER=gcc MODE=tbb
$ make COMPILER=icc MODE=tbb

## CUDA
#### Compilers: nvcc
Version 1 uses unified memory. Version 2 uses explicit memory transfers
$ make COMPILER=nvcc MODE=cuda

## Eigen
#### Compilers: gcc, icc, nvcc
$ make COMPILER=gcc MODE=eigen
$ make COMPILER=icc MODE=eigen
$ make COMPILER=nvcc MODE=eigen

## Alpaka
#### Compilers: gcc, nvcc 
$ make COMPILER=gcc MODE=alpaka
$ make COMPILER=nvcc MODE=alpaka//not yet functional

#### Compile with OpenARC
1. set the environment variable, openarc to the root directory of the OpenARC repository.
2. $ O2GBuild.script
3. $ make COMPILER=openarc MODE=acc SRCTYPE=c
# OpenARC in the public repository has bugs in handling async versions.
# Therefore, do not use propagate-*_async.c files as input.

## Kokkos
The kokkos version exists in its own directory, `src/kokkos_src`, which contains
the p2z code along with a Makefile and README to help the user make use of the
Kokkos library. Here we have basic instructions.

#### Getting started
1. clone https://github.com/kokkos/kokkos to ${HOME}/kokkos
2. in the `src/kokkos_src` direcory edit the Makefile setting to match your needs
3. run `make` to build
4. The executable can be found under `bin` with the others

#### Makefile setting
- KOKKOS_DEVICES = OpenMP or OpenMP,Cuda
- KOKKOS_ARCH = [GPU arch],[CPU arch] i.e. "BDW,Volta70"

