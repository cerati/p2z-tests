# p2z-tests

## OpenMP
#### Compilers: gcc, icc, pgi.
Version 3 is most up to date. 

```shell
$ make COMPILER=gcc MODE=omp
$ make COMPILER=icc MODE=omp
$ make COMPILER=pgi MODE=omp
```

## OpenACC C++
#### Compile with pgi
Compile the OpenACC C++ sync version

```shell
$ make COMPILER=pgi MODE=acc
```

## OpenACC C
#### Compile with pgi, openarc
Compile the OpenACC C sync version

```shell
$ make COMPILER=pgi MODE=accc
$ make COMPILER=openarc MODE=accc
```

Compile the OpenACC C async version (v3)

```shell
$ make COMPILER=pgi MODE=acccv3
$ make COMPILER=openarc MODE=acccv3
```

## OpenMP4 C++
#### Compile with gcc, llvm, ibm
Compile the OpenMP4 C++ sync version

```shell
$ make COMPILER=gcc MODE=omp4
$ make COMPILER=llvm MODE=omp4
$ make COMPILER=ibm MODE=omp4
```

## OpenMP4 C
#### Compile with openarc
Compile the OpenMP4 C sync version

```shell
$ make COMPILER=openarc MODE=omp4c
```

Compile the OpenMP4 C async version (v3)

```shell
$ make COMPILER=openarc MODE=omp4cv3
```

## Tbb
#### Compilers: gcc, icc
```shell
$ make COMPILER=gcc MODE=tbb
$ make COMPILER=icc MODE=tbb
```

## CUDA
#### Compilers: nvcc
Version 1 uses unified memory. Version 2 and 3 use explicit memory transfers.
Version 3 has the same computation and communiation patterns as OpenACC async version (v3).

```shell
$ make COMPILER=nvcc MODE=cuda
$ make COMPILER=nvcc MODE=cudav2
$ make COMPILER=nvcc MODE=cudav3
```

## Eigen
#### Compilers: gcc, icc, nvcc

```shell
$ make COMPILER=gcc MODE=eigen
$ make COMPILER=icc MODE=eigen
$ make COMPILER=nvcc MODE=eigen
```

## Alpaka
#### Compilers: gcc, nvcc 

```shell
$ make COMPILER=gcc MODE=alpaka
$ make COMPILER=nvcc MODE=alpaka//not yet functional
```

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

