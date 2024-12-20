# p2z-tests

## Peer-reviewed publication

Our results are published as "Exploring code portability solutions for HEP with a particle tracking test code", Front. Big Data, 22 October 2024, Sec. Big Data and AI in High Energy Physics, Volume 7 - 2024 (https://doi.org/10.3389/fdata.2024.1485344).

Versions from this repository used in the paper figures are:

* Figure 2,3,4,7:
	* CUDA: [src/propagate-toz-test_CUDA_v4.cu](https://github.com/cerati/p2z-tests/blob/v1.0/src/propagate-toz-test_CUDA_v4.cu)
	* Alpaka: [src/alpaka_src_gpu/src/propagate-toz-test_alpaka_cpu_gpu_v4.cpp](https://github.com/cerati/p2z-tests/blob/v1.0/src/alpaka_src_gpu/src/propagate-toz-test_alpaka_cpu_gpu_v4.cpp)
	* Kokkos: [src/kokkos_src_v6_2/propagate-toz-test_Kokkos_v6_2.cpp](https://github.com/cerati/p2z-tests/blob/v1.0/src/kokkos_src_v6_2/propagate-toz-test_Kokkos_v6_2.cpp)
	* OpenMP4: [src/propagate-toz-test_OpenMP4_async_v4.c](https://github.com/cerati/p2z-tests/blob/v1.0/src/propagate-toz-test_OpenMP4_async_v4.c)
	* OpenACC: [src/propagate-toz-test_Openly_Sung_v4.c](https://github.com/cerati/p2z-tests/blob/v1.0/src/propagate-toz-test_OpenACC_async_v4.c)
	* stdpar: [src/propagate-toz-test_pstl_v2.cpp](https://github.com/cerati/p2z-tests/blob/v1.0/src/propagate-toz-test_pstl_v2.cpp)
 	* TBB: [src/propagate-toz-test_tbb.cpp](https://github.com/cerati/p2z-tests/blob/v1.0/src/src/propagate-toz-test_tbb.cpp)

* Figure 5 (memory prepinning test):
	* OpenACC batched, sync: [src/propagate-toz-test_OpenACC_sync.c](https://github.com/cerati/p2z-tests/blob/v1.0/src/propagate-toz-test_OpenACC_sync.c)
	* OpenACC batched, async: [src/propagate-toz-test_OpenACC_async.c](https://github.com/cerati/p2z-tests/blob/v1.0/src/propagate-toz-test_OpenACC_async.c)
	* OpenACC thread-local, async: [src/propagate-toz-test_OpenACC_async_v4.c](https://github.com/cerati/p2z-tests/blob/v1.0/src/propagate-toz-test_OpenACC_async_v4.c)
	* Kokkos batched, sync: [src/kokkos_src_v3_2/propagate-toz-test_Kokkos_v3_2.cpp](https://github.com/cerati/p2z-tests/blob/v1.0/src/kokkos_src_v3_2/propagate-toz-test_Kokkos_v3_2.cpp)
	* Kokkos batched, async: [src/kokkos_src_v4_2/propagate-toz-test_Kokkos_v4_2.cpp](https://github.com/cerati/p2z-tests/blob/v1.0/src/kokkos_src_v4_2/propagate-toz-test_Kokkos_v4_2.cpp)
	* Kokkos thread-local, sync: [src/kokkos_src_v5_2/propagate-toz-test_Kokkos_v5_2.cpp](https://github.com/cerati/p2z-tests/blob/v1.0/src/kokkos_src_v5_2/propagate-toz-test_Kokkos_v5_2.cpp)
	* Kokkos thread-local, async: [src/kokkos_src_v6_2/propagate-toz-test_Kokkos_v6_2.cpp](https://github.com/cerati/p2z-tests/blob/v1.0/src/kokkos_src_v6_2/propagate-toz-test_Kokkos_v6_2.cpp)

Please find them as used in the paper at the tagged version of this repository: https://github.com/cerati/p2z-tests/releases/tag/v1.0. 

## OpenMP
#### Compilers: gcc, icc, nvhpc.
Version 3 is most up to date. 

```shell
$ make COMPILER=gcc MODE=omp
$ make COMPILER=icc MODE=omp
$ make COMPILER=nvhpc MODE=omp
```

## OpenACC C
#### Compile with nvhpc, openarc, gcc
Compile the OpenACC C sync version, which may use CUDA shared memory for temporary data.
(In OpenACC, different OpenACC compilers may allocate the temporary data on different memory spaces.)

The computation and communication patterns of the acccv3 version is logically equivalent to CUDA version 3 (cudav3).

The accc version is a synchronous version of the acccv3 version.

The computation and communication patterns of the acccv4 version is logically equivalent to CUDA version 4 (cudav4).

```shell
$ make COMPILER=nvhpc MODE=accc INCLUDE_DATA=1 USE_FMAD=1 #NVHPC V22.11 compiles correctly.
$ make COMPILER=openarc MODE=accc INCLUDE_DATA=1 USE_FMAD=1
$ make COMPILER=gcc MODE=accc INCLUDE_DATA=1
```
Enable the following environment variable to make OpenARC-compiled program generate the same outputs

```shell
export OPENARC_JITOPTION="--fmad false"
```

Compile the OpenACC C async version (v3 and v4)

```shell
$ make COMPILER=nvhpc MODE=acccv3 INCLUDE_DATA=1 USE_FMAD=1 #NVHPC V22.11 compiles correctly.
$ make COMPILER=nvhpc MODE=acccv4 INCLUDE_DATA=1 USE_FMAD=1 #NVHPC V22.11 compiles correctly.
$ make COMPILER=openarc MODE=acccv3 INCLUDE_DATA=1 USE_FMAD=1
$ make COMPILER=openarc MODE=acccv4 INCLUDE_DATA=1 USE_FMAD=1
$ make COMPILER=gcc MODE=acccv3 INCLUDE_DATA=1
$ make COMPILER=gcc MODE=acccv4 INCLUDE_DATA=1
```

## OpenACC C for CPU
#### Compile with nvhpc, openarc
Compile the OpenACC C sync version for CPU

```shell
$ make COMPILER=nvhpc MODE=accccpu
$ make COMPILER=openarc MODE=accccpu
$ make COMPILER=gcc MODE=accccpu
```

Compile the OpenACC C async version (v3 and v4) for CPU

```shell
$ make COMPILER=nvhpc MODE=acccv3cpu INCLUDE_DATA=0
$ make COMPILER=nvhpc MODE=acccv4cpu INCLUDE_DATA=0
$ make COMPILER=openarc MODE=acccv3cpu
$ make COMPILER=openarc MODE=acccv4cpu
$ make COMPILER=gcc MODE=acccv3cpu 
$ make COMPILER=gcc MODE=acccv4cpu 
```

## OpenMP4 C
#### Compile with openarc, llvm, ibm, and gcc
Compile the OpenMP4 C sync version

The computation and communication patterns of the omp4cv3 version is logically equivalent to CUDA version 3 (cudav3).

The omp4c version is a synchronous version of the omp4cv3 version.

The computation and communication patterns of the omp4cv4 version is logically equivalent to CUDA version 4 (cudav4).

```shell
$ make COMPILER=openarc MODE=omp4c INCLUDE_DATA=1 USE_FMAD=1
$ make COMPILER=llvm MODE=omp4c INCLUDE_DATA=1 #LLVM V15.0.0
$ make COMPILER=ibm MODE=omp4c INCLUDE_DATA=1 #XLC V16.1.1-10
$ make COMPILER=gcc MODE=omp4c INCLUDE_DATA=1 #GCC V12.2.1
$ make COMPILER=nvhpc MODE=omp4c INCLUDE_DATA=1 USE_FMAD=1 #NVHPC V21.11 fails due to an unsupported feature (Standalone 'omp parallel' in a 'declare target' routine is not supported yet). NVHPC V22.11 also fails.
```

Compile the OpenMP4 C async version (v3 and v4)

```shell
$ make COMPILER=openarc MODE=omp4cv3 INCLUDE_DATA=1 USE_FMAD=1
$ make COMPILER=openarc MODE=omp4cv4 INCLUDE_DATA=1 USE_FMAD=1
$ make COMPILER=llvm MODE=omp4cv3 INCLUDE_DATA=1
$ make COMPILER=llvm MODE=omp4cv4 INCLUDE_DATA=1
$ make COMPILER=ibm MODE=omp4cv3 INCLUDE_DATA=1
$ make COMPILER=ibm MODE=omp4cv4 INCLUDE_DATA=1
$ make COMPILER=gcc MODE=omp4cv3 INCLUDE_DATA=1
$ make COMPILER=gcc MODE=omp4cv4 INCLUDE_DATA=1
$ make COMPILER=nvhpc MODE=omp4cv3 INCLUDE_DATA=1 USE_FMAD=1 #NVHPC V21.11 fails due to an unsupported feature (Standalone 'omp parallel' in a 'declare target' routine is not supported yet). NVHPC V22.11 also fails.
$ make COMPILER=nvhpc MODE=omp4cv4 INCLUDE_DATA=1 USE_FMAD=1 #NVHPC V22.11 compiles correctly.
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
Version 0 (cuda) has the same computation and communication patterns as version 3 (cudav3) but uses unified memory.

Version 1 (cudav1) has the same computation and communication patterns as version 4 (cudav4) but uses unified memory.

Version 2 (cudav2) uses explicit memory transfers, and each device thread has a local copy of each batched temporary track data (e.g., struct MP6x6F errorProp), even though each thread needs to access only assigned part of the batched track data, which is inefficient in terms of device memory usage.

Version 3 has the same computation and communication patterns as version 2 (cudav2), but device threads in the same thread block share the batched temporary track data by allocating the batched data in the CUDA shared memory (e.g., __shared__ struct MP6x6F errorProp).
Version 3 has the same computation and communication patterns as OpenACC async version (acccv3).

Version 4 has the same computation and communication patterns as version 2 (cudav2), but each device threads stores only assigned temporary track data in its thread-private memory (i.e., CUDA local memory), instead of using CUDA shared memory.
Version 4 has the same computation and communication patterns as OpenACC async version (acccv4).

The computation and communication patterns of the CUDAUVM version (cudauvm) is logically equivalent to CUDA version 4 (cudav4); the main difference between cudauvm and cudav4 is that cudauvm is written in a typical C++ style (e.g., use C++ template struct) while cudav4 is written in a C style (e.g., use C struct).

```shell
$ make COMPILER=nvcc MODE=cuda #default behaviors: asynchronous execution; measure both memory transfer times and compute times
$ make COMPILER=nvcc MODE=cuda INCLUDE_DATA=0 #measure compute times only
$ make COMPILER=nvcc MODE=cuda USE_FMAD=0 #disable the fmad optimization
$ make COMPILER=nvcc MODE=cuda INCLUDE_DATA=1 USE_FMAD=1 #default
$ make COMPILER=nvcc MODE=cudav1 USE_ASYNC=1 #for asynchronous execution (default)
$ make COMPILER=nvcc MODE=cudav1 USE_ASYNC=0 #for synchronous execution
$ make COMPILER=nvcc MODE=cudav1 INCLUDE_DATA=1 USE_FMAD=1 USE_ASYNC=1 #default
$ make COMPILER=nvcc MODE=cudav2 INCLUDE_DATA=1 USE_FMAD=1 USE_ASYNC=1 #default
$ make COMPILER=nvcc MODE=cudav3 INCLUDE_DATA=1 USE_FMAD=1 USE_ASYNC=1 #default
$ make COMPILER=nvcc MODE=cudav4 INCLUDE_DATA=1 USE_FMAD=1 USE_ASYNC=1 #default
$ make COMPILER=nvcc MODE=cudauvm INCLUDE_DATA=1 USE_FMAD=1 #default
```

## CUDA Hybrid
#### Compilers: nvc++
GCC_ROOT option has to be set properly to use a specific GCC version

The hybrid version (cudahyb) mixes CUDA and C++ parallel algorithm for GPU offloading.

The computation and communication patterns of this hybrid version (cudahyb) is logically equivalent to CUDA version 4 (cudav4).

```shell
$ make COMPILER=nvhpc MODE=cudahyb INCLUDE_DATA=1 USE_FMAD=1 #defaul
$ make COMPILER=nvhpc MODE=cudahyb USE_FMAD=0 #disable the fmad optimization
$ make COMPILER=nvhpc MODE=cudahyb INCLUDE_DATA=0 #measure compute times only
```

## PSTL
#### Compilers: nvc++
GCC_ROOT option has to be set properly to use a specific GCC version

The PSTL version (pstl) uses C++ parallel algorithm for GPU offloading.

The computation and communication patterns of this PSTL version (pstl) is logically equivalent to CUDA version 4 (cudav4).

```shell
$ make COMPILER=nvhpc MODE=pstl INCLUDE_DATA=1 USE_FMAD=1 #default
$ make COMPILER=nvhpc MODE=pstl USE_FMAD=0 #disable the fmad optimization
$ make COMPILER=nvhpc MODE=pstl INCLUDE_DATA=0 #measure compute times only
```

## Eigen
#### Compilers: gcc, icc, nvcc
The Eigen version is outdated (e.g., KalmanUpdate() should be updated.)

The computation and communication patterns of this Eigen version (eigen) is logically similar to CUDA version 2 (cudav2).

```shell
$ make COMPILER=gcc MODE=eigen
$ make COMPILER=icc MODE=eigen
$ make COMPILER=nvcc MODE=eigen
```

## Alpaka
#### Compilers: gcc, nvcc 
ALPAKA_INSTALL_ROOT should be set to the Alpaka install root directory.
(Refer to ./src/alpaka_src_gpu/Readme.md to install Alpaka.)

The computation and communication patterns of this Alpaka version (alpaka) is logically equivalent to CUDA version 3 (cudav3).
The computation and communication patterns of this Alpaka version 4 (alpakav4) is logically equivalent to CUDA version 4 (cudav4).

```shell
$ make COMPILER=nvcc MODE=alpaka #compile src/alpaka_src_gpu/src/propagate-toz-test_alpaka_cpu_gpu.cpp for GPU using nvcc
$ make COMPILER=nvcc MODE=alpaka USE_FMAD=0 #disable fmad optimization
$ make COMPILER=nvcc MODE=alpaka USE_FMAD=1 INCLUDE_DATA=1 
$ make COMPILER=nvcc MODE=alpakav4 USE_FMAD=1 INCLUDE_DATA=1 
$ make COMPILER=gcc MODE=alpaka #compile src/alpaka_src_gpu/src/propagate-toz-test_alpaka_cpu_gpu.cpp for CPU using gcc
$ make COMPILER=gcc MODE=alpaka ALPAKASRC=. #compile src/propagate-toz-test_alpaka.cpp for CPU using gcc
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
                  CUDA version 3 (cudav3)(`propagate-toz-test_CUDA_v3.cu`) but using 
                  a single device instance.

`src/kokkos_src_v3_2`: is a variant of `src/kokkos_src_v3`, where inner parallel_for
                  constructs are manually fused.

`src/kokkos_src_v4`: has the same user data layouts and compute patterns 
                  as `src/kokkos_src_v3, but use multiple asynchronous device 
                  instances (multiple CUDA streams)

`src/kokkos_src_v4_2`: is a variant of `src/kokkos_src_v4`, where inner parallel_for
                  constructs are manually fused.

`src/kokkos_src_v5`: target CUDA GPU without unified memory; has the same 
                  user data layouts and memory transfer patterns as the manual
                  CUDA version 4 (cudav4)(`propagate-toz-test_CUDA_v4.cu`) but using 
                  a single device instance.

`src/kokkos_src_v5_2`: is a variant of `src/kokkos_src_v5`, where inner parallel_for
                  constructs are manually fused.

`src/kokkos_src_v6`: has the same user data layouts and compute patterns 
                  as `src/kokkos_src_v5, but use multiple asynchronous device 
                  instances (multiple CUDA streams).

`src/kokkos_src_v6_2`: is a variant of `src/kokkos_src_v6`, where inner parallel_for
                  constructs are manually fused; the new best performing version.

`src/kokkos_src_v7`: has the same user data layouts and compute patterns as the P2R Kokkos version.

Here we have basic instructions.

#### Getting started
1. clone https://github.com/kokkos/kokkos to ${KOKKOS_ROOT}
2. compile option 1:
	a. in the `src/kokkos_src_v6` directory edit the Makefile setting to match your needs
	b. run `make` to build
	c. The executable can be found under `bin` with the others
3. compile option 2:
	compile using the Makefile in the current directory:
	(KOKKOS_ROOT needs to be set to the Kokkos root directory)
		- use KOKKOS_ARCH option to set a target device architecture (default GPU architecture: Volta70, CPU architecture: SKX)

```shell
$ make COMPILER=gcc MODE=kokkosv1 USE_GPU=0 KOKKOS_ARCH=BDW #compile Kokkos V1 for Intel Broadwell Xeon CPU using OpenMP
$ make COMPILER=nvcc MODE=kokkosv1 USE_FMAD=1 USE_GPU=1 INCLUDE_DATA=1 #compile Kokkos V1 for NVIDIA Volta GPU using CUDA
$ make COMPILER=nvcc MODE=kokkosv2 INCLUDE_DATA=1 USE_FMAD=1 USE_GPU=1
$ make COMPILER=gcc MODE=kokkosv2 INCLUDE_DATA=0 USE_GPU=0 KOKKOS_ARCH=BDW
$ make COMPILER=nvcc MODE=kokkosv3 INCLUDE_DATA=1 USE_FMAD=1 USE_GPU=1
$ make COMPILER=gcc MODE=kokkosv3 INCLUDE_DATA=0 USE_GPU=0 KOKKOS_ARCH=BDW
$ make COMPILER=nvcc MODE=kokkosv3 INCLUDE_DATA=1 USE_FMAD=1 USE_GPU=1 PREPIN_HOSTMEM=1 #work only for NVIDIA GPUs
$ make COMPILER=nvcc MODE=kokkosv4 INCLUDE_DATA=1 USE_FMAD=1 USE_GPU=1 #work only for NVIDIA GPUs
$ make COMPILER=nvcc MODE=kokkosv4 INCLUDE_DATA=1 USE_FMAD=1 USE_GPU=1 PREPIN_HOSTMEM=1 STREAMS=1 #work only for NVIDIA GPUs
$ make COMPILER=nvcc MODE=kokkosv5 INCLUDE_DATA=1 USE_FMAD=1 USE_GPU=1
$ make COMPILER=gcc MODE=kokkosv5 INCLUDE_DATA=0 USE_GPU=0 KOKKOS_ARCH=BDW
$ make COMPILER=gcc MODE=kokkosv5 INCLUDE_DATA=0 USE_GPU=0 KOKKOS_ARCH=HSW
$ make COMPILER=nvcc MODE=kokkosv5 INCLUDE_DATA=1 USE_FMAD=1 USE_GPU=1 PREPIN_HOSTMEM=1 #work only for NVIDIA GPUs
$ make COMPILER=nvcc MODE=kokkosv6 INCLUDE_DATA=1 USE_FMAD=1 USE_GPU=1 #work only for NVIDIA GPUs
$ make COMPILER=nvcc MODE=kokkosv6 INCLUDE_DATA=1 USE_FMAD=1 USE_GPU=1 PREPIN_HOSTMEM=1 STREAMS=1 #work only for NVIDIA GPUs
$ make COMPILER=nvcc MODE=kokkosv6 INCLUDE_DATA=0 USE_FMAD=1 USE_GPU=1 PREPIN_HOSTMEM=1 #work only for NVIDIA GPUs
$ make COMPILER=nvcc MODE=kokkosv6_2 INCLUDE_DATA=0 USE_FMAD=1 USE_GPU=1 PREPIN_HOSTMEM=1 #work only for NVIDIA GPUs
$ make COMPILER=nvcc MODE=kokkosv5_2 INCLUDE_DATA=0 USE_FMAD=1 USE_GPU=1 KOKKOS_ARCH=AMPERE86
$ make COMPILER=nvcc MODE=kokkosv7 INCLUDE_DATA=0 USE_FMAD=1 USE_GPU=1 
$ make COMPILER=gcc MODE=kokkosv7 USE_GPU=0 KOKKOS_ARCH=POWER9

$ make MODE=kokkosv6 clean
```

#### Makefile setting
- KOKKOS_DEVICES = Cuda or OpenMP or OpenMP,Cuda
- KOKKOS_ARCH = [GPU arch],[CPU arch] i.e. "BDW,Volta70"

