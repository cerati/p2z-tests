# Kokkos Instructions

This directory contains the kokkos version and Makefile for the p2z benchmark.

To use kokkos you must first download a copy of the kokkos source code. The 
library is built each time you build your application with the settings you
choose for the kokkos data structure and parallel 

kokkos source: https://github.com/kokkos/kokkos

## Building Kokkos P2Z

To build simply run `make -j`. There are options detailed below.

Currently, our version of Kokkos P2Z can be adjusted in several parts of the 
`Makefile`. These changes will not force a rebuild of kokkos but often require
one, so use `make clean` to ensure that changes are included in the new binary.
Also, use `-j` since build the kokkos library takes time.

Makefile options:
* KOKKOS_PATH - where the above src is downloaded (default: ${KOKKOS_ROOT})
* KOKKOS_DEVICES - type of parallelism: `Cuda` for GPU
                   (default: Cuda)
* KOKKOS_ARCH - type of hardware: `SKX` means skylake add Volta70 for V100
                (default: Volta70)
  * the Cuda option builds with -DUSE_GPU which is used in the h files
* CXX - compiler 
  * use the nvcc wrapper for cuda and your preference for CPU (g++ 9 recommended)
    (default: ${KOKKOS_PATH}/bin/nvcc_wrapper)
* NITER - the number of iterations (default: 10)
* NLAYER - the number of layers (default: 20)
* num_streams - the number of streams (default: 10)
* prepin_hostmem - set to 1 to prepin host memory (default)
                   set to 0 to disable host memory prepinning
* INCLUDE_DATA - decide whether to include the memory transfer times or not for profiling (default: 1)
