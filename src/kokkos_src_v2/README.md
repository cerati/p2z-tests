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
* KOKKOS_DEVICES - type of parallelism: `OpenMP` for CPU; `Cuda` for GPU
                   (default: Cuda)
* KOKKOS_ARCH - type of hardware: `SKX` means skylake add Volta70 for V100
                (default: Volta70)
  * the Cuda option builds with -DUSE_GPU which is used in the h files
* CXX - compiler 
  * use the nvcc wrapper for cuda and your preference for CPU (g++ 9 recommended)
    (default: ${KOKKOS_PATH}/bin/nvcc_wrapper)
* NITER - the number of iterations (default: 10)
* NLAYER - the number of layers (default: 20) 

kokkos config header file options
* lines 53 through 63 contain the settings for GPU and CPU data structures which 
should be automatically selected based on the KOKKOS_DEVICES option in the Makefile
* These set how the data is stored and processed
*  ExecSpace - The backend execution method (OpenMP and CUDA have been tested)
*  MemSpace - The where the memory is allocated (Cuda, CudaUVM, OpenMP, Host)
  * Cuda - requires manual data transfers 
  * OpenMP / Host - On the CPU difference is unclear
* Layout - row- or column- wise data layouts
  * LayoutRight - Row major, for CPU, with logically adjacent element adjacent in memory
  * LayoutLeft - Column major for GPU, with logically adjacent element adjacent in memory




