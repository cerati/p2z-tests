#!/bin/bash

source ../../mkFit/xeon_scripts/init-env.sh

#rm tuner_output.txt

#trktunes=(9600 12800) 
trktunes=(7168 8192 9216 10240 11264 12288) #1024*(7-12)
#trktunes=(9600 12800 19200 38400 64000 96000)
#tunes=(1 5 10 15 20 25 30 35 40 45 50 55 60 65 70 75 80 85 90 95 100 105 110 115 120 125 130 135 140 145 150)

#btunes=(1 2 4 8 16 32 64 128)
btunes=(1 2 4 8 16 32 64 128 256 512 1024)
events=(1 10 50 100 250 500 750 1000)
compilers=("gcc" "icc" "pgi" "gcc" "icc" "gcc"   "icc"  ) #only tunable cpu versions right now
    modes=("omp" "omp" "omp" "tbb" "tbb" "eigen" "eigen")
#compilers=("gcc" "icc" "pgi" "pgi" "gcc" "icc" "gcc"   "icc"   "gcc"    "icc"    "nvcc"   "nvcc")
#    modes=("omp" "omp" "omp" "acc" "tbb" "tbb" "eigen" "eigen" "alpaka" "alpaka" "alpaka" "cuda")
#compilers=("pgi")
#modes=("omp")
nthreads=(1 8 16 32 64 128 256)


for idx in "${!compilers[@]}";do
  compiler=${compilers[$idx]}
  mode=${modes[$idx]}
  filename="../bin/propagate_${compiler}_${mode}"
  outname="output_tuner/output_${compiler}_${mode}.txt"
  echo ${compiler} and ${mode}
  rm ${outname}
  
for event in ${events[@]}; do 
  for trktune in ${trktunes[@]}; do
    for btune in ${btunes[@]}; do
      for nthread in ${nthreads[@]}; do 
        make -C .. COMPILER=${compiler} MODE=${mode} TUNEB=${btune} TUNETRK=${trktune} TUNEEVT=${event} NTHREADS=${nthread} clean
        make -C .. COMPILER=${compiler} MODE=${mode} TUNEB=${btune} TUNETRK=${trktune} TUNEEVT=${event} NTHREADS=${nthread}
        for ((i=0; i<5; i++)); do
          #echo "${mode}(${compiler})" ${compiler} ${mode} ${btune} ${trktune} ${i}  
          #echo -n "${mode}(${compiler})" ${compiler} ${mode} ${btune} ${trktune} ${i}' ' >> ${outname}
          echo "${mode}(${compiler})" ${compiler} ${mode} ${btune} ${trktune} ${event} ${nthread} ${i} 
          echo -n "${mode}(${compiler})" ${compiler} ${mode} ${i}' ' >> ${outname}
          ./${filename} | grep "formatted" | cut -d' ' -f2- >> ${outname}
        done
      done
    done
  done
done
done




