# Kokkos Instructions

To use kokkos you must first download a copy of the source as it will build 
itself depending on your choice of devices.

kokkos source: https://github.com/kokkos/kokkos

## Building Kokkos P2Z

Currently, our version of Kokkos P2Z can be adjusted in several parts of the 
`Makefile` and in the `ptoz_data.h` file. Changes to the `.h` will not force
 a rebuild but often require one, so use `make clean` to ensure that changes
happen. Also, use `-j` since build the kokkos library takes time.

Makefile options:
* KOKKOS_PATH - where the above src is downloaded
* KOKKOS_DEVICES - type of parallelism: `OpenMP` for cpu; `OpenMP,Cuda` fpr gpu
* KOKKOS_ARCH - type of hardware: `BDW` means broadwell add Volta70 for V100
  * a skylake keyword maybe be available now
* CXX - compiler 
  * use the nvcc wrapper for cuda and your preference for cpu (g++ 9 recommended)

h file options
* ptoz MemSpace_CB (line about 72) - should be switch to UVM for the GPU version
* kokkos config ExeSpace should be switch between OpenMP and Cuda as necessary
* kokkos config Layout should be Right for CPU and left for GPU
  * more testing should be done to compare these however

