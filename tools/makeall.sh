#!/bin/bash

source ../../mkFit/xeon_scripts/init-env.sh

make -C .. cleanall
compilers=("gcc" "icc" "pgi" "pgi" "gcc" "icc" "gcc"   "icc"   "nvcc"  "gcc"    "icc"    "nvcc"   "nvcc" "nvcc")
    modes=("omp" "omp" "omp" "acc" "tbb" "tbb" "eigen" "eigen" "eigen" "alpaka" "alpaka" "alpaka" "cuda" "cudav2")


for idx in "${!compilers[@]}";do
  compiler=${compilers[$idx]}
  mode=${modes[$idx]}
  echo ${compiler} and ${mode}
  
 # make -C .. COMPILER=${compiler} MODE=${mode} TUNEB=${btune} TUNETRK=${trktune} TUNEEVT=${event} NTHREADS=${nthread} clean
  make -C .. COMPILER=${compiler} MODE=${mode} NITER="100" 
done




