# p2z-tests

## OpenMP
#### Compilers: gcc, icc, pgi.
Version 3 is most up to date. 

```shell
$ make COMPILER=gcc MODE=omp
$ make COMPILER=icc MODE=omp
$ make COMPILER=pgi MODE=omp
```

## OpenACC C
#### Compile with pgi, openarc, gcc
Compile the OpenACC C sync version

```shell
$ make COMPILER=pgi MODE=accc
$ make COMPILER=openarc MODE=accc INCLUDE_DATA=0
$ make COMPILER=gcc MODE=accc
```
Enable the following environment variable to make OpenARC-compiled program generate the same outputs

```shell
export OPENARC_JITOPTION="--fmad false"
```

Compile the OpenACC C async version (v3 and v4)

```shell
$ make COMPILER=pgi MODE=acccv3
$ make COMPILER=openarc MODE=acccv3 INCLUDE_DATA=0
$ make COMPILER=gcc MODE=acccv3
$ make COMPILER=openarc MODE=acccv4 INCLUDE_DATA=0
```

## OpenACC C for CPU
#### Compile with pgi, openarc
Compile the OpenACC C sync version for CPU

```shell
$ make COMPILER=pgi MODE=accccpu
$ make COMPILER=openarc MODE=accccpu
$ make COMPILER=gcc MODE=accccpu
```

Compile the OpenACC C async version (v3 and v4) for CPU

```shell
$ make COMPILER=pgi MODE=acccv3cpu
$ make COMPILER=openarc MODE=acccv3cpu
$ make COMPILER=gcc MODE=acccv3cpu
$ make COMPILER=openarc MODE=acccv4cpu
```

## OpenMP4 C
#### Compile with openarc, llvm, ibm, and gcc
Compile the OpenMP4 C sync version

```shell
$ make COMPILER=openarc MODE=omp4c INCLUDE_DATA=0
$ make COMPILER=llvm MODE=omp4c
$ make COMPILER=ibm MODE=omp4c
$ make COMPILER=gcc MODE=omp4c
$ make COMPILER=pgi MODE=omp4c #NVHPC V21.11 fails due to an unsupported feature (Standalone 'omp parallel' in a 'declare target' routine is not supported yet). NVHPC V22.2 also fails.
```

Compile the OpenMP4 C async version (v3 and v4)

```shell
$ make COMPILER=openarc MODE=omp4cv4 INCLUDE_DATA=0
$ make COMPILER=openarc MODE=omp4cv3 INCLUDE_DATA=0
$ make COMPILER=llvm MODE=omp4cv3
$ make COMPILER=ibm MODE=omp4cv3
$ make COMPILER=gcc MODE=omp4cv3
$ make COMPILER=pgi MODE=omp4cv3 #NVHPC V21.11 fails due to an unsupported feature (Standalone 'omp parallel' in a 'declare target' routine is not supported yet). NVHPC V22.2 also fails.
```

Enable the following environment variable to make OpenARC-compiled program generate the same outputs

```shell
export OPENARC_JITOPTION="--fmad false"
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
$ make COMPILER=nvcc MODE=cuda #default behaviors: asynchronous execution; measure both memory transfer times and compute times
$ make COMPILER=nvcc MODE=cuda INCLUDE_DATA=1 #measure both memory transfer times and compute times (default)
$ make COMPILER=nvcc MODE=cuda INCLUDE_DATA=0 #measure compute times only
$ make COMPILER=nvcc MODE=cuda USE_FMAD=0 #disable the fmad optimization
$ make COMPILER=nvcc MODE=cudav1 USE_ASYNC=1 #for asynchronous execution (default)
$ make COMPILER=nvcc MODE=cudav1 USE_ASYNC=0 #for synchronous execution
$ make COMPILER=nvcc MODE=cudav1 INCLUDE_DATA=1 #measure both memory transfer times and compute times (default)
$ make COMPILER=nvcc MODE=cudav1 INCLUDE_DATA=0 #measure compute times only
$ make COMPILER=nvcc MODE=cudav2 INCLUDE_DATA=1 #measure both memory transfer times and compute times (default)
$ make COMPILER=nvcc MODE=cudav2 INCLUDE_DATA=0 #measure compute times only
$ make COMPILER=nvcc MODE=cudav3 INCLUDE_DATA=1 #measure both memory transfer times and compute times (default)
$ make COMPILER=nvcc MODE=cudav3 INCLUDE_DATA=0 #measure compute times only
$ make COMPILER=nvcc MODE=cudav4 USE_ASYNC=1 #for asynchronous execution (default)
$ make COMPILER=nvcc MODE=cudav4 USE_ASYNC=0 #for synchronous execution
$ make COMPILER=nvcc MODE=cudav4 INCLUDE_DATA=1 #measure both memory transfer times and compute times (default)
$ make COMPILER=nvcc MODE=cudav4 INCLUDE_DATA=0 #measure compute times only
$ make COMPILER=nvcc MODE=cudauvm INCLUDE_DATA=1 #measure both memory transfer times and compute times (default)
$ make COMPILER=nvcc MODE=cudauvm INCLUDE_DATA=0 #measure compute times only
```

## CUDA Hybrid
#### Compilers: nvc++
GCC_ROOT option has to be set properly to use a specific GCC version

```shell
$ make COMPILER=nvcpp MODE=cudahyb 
$ make COMPILER=nvcpp MODE=cudahyb USE_FMAD=0 #disable the fmad optimization
$ make COMPILER=nvcpp MODE=cudahyb INCLUDE_DATA=0 #measure compute times only
```

## PSTL
#### Compilers: nvc++
GCC_ROOT option has to be set properly to use a specific GCC version

```shell
$ make COMPILER=nvcpp MODE=pstl 
$ make COMPILER=nvcpp MODE=pstl USE_FMAD=0 #disable the fmad optimization
$ make COMPILER=nvcpp MODE=pstl INCLUDE_DATA=0 #measure compute times only
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
Multiple Kokkos versions exist in the `src` directory: `src/kokkos_src_v1`, 
..., `src/kokkos_src_v6`, each of which contains the p2z code along with 
a Makefile and README to help the user make use of the Kokkos library. 

Here is the brief information on each version:

`src/kokkos_src_v1`: target CUDA GPU with unified memory; has user data layouts
                  different from those in the manual CUDA versions.

`src/kokkos_src_v2`: target CUDA GPU without unified memory; has the same 
                  user data layouts as `src/kokkos_src_v1` but with explicit
                  memory transfers at each iteration of the outermost `itr` loop,
                  as the manual CUDA versions do.

`src/kokkos_src_v3`: target CUDA GPU without unified memory; has the same 
                  user data layouts and memory transfer patterns as the manual
                  CUDA_v3 version (`propagate-toz-test_CUDA_v3.cu`) but using 
                  a single device instance.

`src/kokkos_src_v4`: has the same user data layouts and compute patterns 
                  as `src/kokkos_src_v3, but use multiple asynchronous device 
                  instances (multiple CUDA streams)

`src/kokkos_src_v5`: target CUDA GPU without unified memory; has the same 
                  user data layouts and memory transfer patterns as the manual
                  CUDA_v4 version (`propagate-toz-test_CUDA_v4.cu`) but using 
                  a single device instance.

`src/kokkos_src_v6`: has the same user data layouts and compute patterns 
                  as `src/kokkos_src_v5, but use multiple asynchronous device 
                  instances (multiple CUDA streams); the best performing version.

Here we have basic instructions.

#### Getting started
1. clone https://github.com/kokkos/kokkos to ${KOKKOS_ROOT}
2. compile option 1:
	a. in the `src/kokkos_src_v6` direcory edit the Makefile setting to match your needs
	b. run `make` to build
	c. The executable can be found under `bin` with the others
3. compile option 2:
	compile using the Makefile in the current directory:

```shell
$ make COMPILER=nvcc MODE=kokkosv1 INCLUDE_DATA=0
$ make COMPILER=nvcc MODE=kokkosv2 INCLUDE_DATA=0
$ make COMPILER=nvcc MODE=kokkosv3 INCLUDE_DATA=0
$ make COMPILER=nvcc MODE=kokkosv4 INCLUDE_DATA=0
$ make COMPILER=nvcc MODE=kokkosv5 INCLUDE_DATA=0
$ make COMPILER=nvcc MODE=kokkosv6 INCLUDE_DATA=0
$ make MODE=kokkosv6 clean
```

#### Makefile setting
- KOKKOS_DEVICES = OpenMP or OpenMP,Cuda
- KOKKOS_ARCH = [GPU arch],[CPU arch] i.e. "BDW,Volta70"

