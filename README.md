# p2z-tests

## OpenMP
TODO


## OpenACC
#### Compile with PGI
//Compile OpenACC C++ version
$ make COMPILER=pgi MODE=acc SRCTYPE=cpp
//Compile OpenACC C version
$ make COMPILER=pgi MODE=acc SRCTYPE=c

#### Compile with OpenARC
1. set the environment variable, openarc to the root directory of the OpenARC repository.
2. $ O2GBuild.script
3. $ make COMPILER=openarc MODE=acc SRCTYPE=c

## Kokkos

#### Getting started
1. clone https://github.com/kokkos/kokkos to ${HOME}/kokkos
2. make will build everything
3. ./test.cuda to run

#### Makefile setting
- KOKKOS_DEVICES = OpenMP or OpenMP,Cuda
- KOKKOS_ARCH = [GPU arch],[CPU arch] i.e. "BDW,Volta70"
- 

