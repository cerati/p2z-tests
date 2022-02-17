## REQUIREMENTS

- Install OpenARC compiler

First, set up the environment variable, `openarc` to the root directory of the OpenARC repository.

```shell
$ export openarc=[OpenARC-root-directory]
```

Second, set up the environment variable, `OPENARC_ARCH` to a correct value depending on the target device.
(See README.md file in the OpenARC repository for more details.)

```shell
$ export OPENARC_ARCH=0 #for NVIDIA GPUs with CUDA.

$ export OPENARC_ARCH=1 #for general OpenCL devices.

$ export OPENARC_ARCH=2 #for Xeon Phi with OpenCL 

$ export OPENARC_ARCH=3 #for Intel FPGA with OpenCL

$ export OPENARC_ARCH=4 #for MCL with OpenCL

$ export OPENARC_ARCH=5 #for AMD GPUs with HIP

$ export OPENARC_ARCH=6 #for IRIS/Brisbane
```

Third, build the OpenARC compiler.

```shell
$ cd ${openarc}
$ make purge
$ make
```

## COMPILE and RUN P2Z BENCHMARK
Environment variables, `OPENARC_ARCH` and `openarc` should be set up properly to run the build scripts correctly.

- To compile and run OpenACC C sync version of P2Z

```shell
$ cd [directory-where-this-file-exists]
$ make COMPILER=openarc MODE=accc
$ make
$ cd ./bin
$ ./propagate_openarc_accc
```

- To compile OpenACC C async version (v3) of P2Z

```shell
$ make COMPILER=openarc MODE=acccv3
```

- To compile OpenMP4 C sync version of P2Z

```shell
$ make COMPILER=openarc MODE=omp4c
```
- To compile OpenMP4 C async version (v3) of P2Z

```shell
$ make COMPILER=openarc MODE=omp4cv3
```

- To compile OpenACC C sync version of P2Z for CPU

```shell
$ make COMPILER=openarc MODE=accccpu
```

- To compile OpenACC C async version (v3) of P2Z for CPU

```shell
$ make COMPILER=openarc MODE=acccv3cpu
```

- To enable the host-memory-prepinning optimization, set up the environment variable, `OPENARCRT_PREPINHOSTMEM` to `1` before running the application.

```shell
$ export OPENARCRT_PREPINHOSTMEM=1
```
